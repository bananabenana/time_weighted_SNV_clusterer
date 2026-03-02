#!/usr/bin/env python3
import argparse
import math
import subprocess
from itertools import combinations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
import csv
import networkx as nx
import polars as pl
import tempfile


def parse_manifest_and_rates(manifest_file, cluster_snvs_file=None):
    clusters = defaultdict(list)
    genome_info = {}
    cluster_rates = {}

    print(f"Loading manifest file: {manifest_file}")
    with open(manifest_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"Genome", "Path", "Predefined_lineage_cluster", "Date"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{manifest_file} must contain header: Genome, Path, Predefined_lineage_cluster, Date"
            )

        for row in reader:
            genome = row["Genome"]
            path = row["Path"]
            lineage = row["Predefined_lineage_cluster"].strip()
            date = row["Date"]

            if not lineage:
                print(f"WARNING: {genome} contained no Predefined_lineage_cluster in {manifest_file}. No cluster assigned.")
                genome_info[genome] = {"cluster_id": None, "path": path, "isolation_date": date}
                continue

            clusters[lineage].append((genome, path, date))
            genome_info[genome] = {"cluster_id": lineage, "path": path, "isolation_date": date}

    if cluster_snvs_file:
        print(f"Loading lineage-specific SNVs per year file: {cluster_snvs_file}")
        with open(cluster_snvs_file) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cluster_rates[row["Predefined_lineage_cluster"].strip()] = float(row["Predefined_SNVs_per_year"])

    print(f"Loaded {len(genome_info)} genomes, {len(clusters)} lineages with >0 genomes.")
    return clusters, genome_info, cluster_rates


def get_genome_sizes(genome_info):
    print("Calculating genome sizes...")
    sizes = {
        g: sum(len(line.strip()) for line in open(info["path"]) if not line.startswith(">") and line.strip()) / 1e6
        for g, info in genome_info.items() if info["cluster_id"] is not None
    }
    print("Genome sizes calculated.")
    return sizes


def run_mash_distance(args):
    g1, g2, path1, path2 = args
    with tempfile.NamedTemporaryFile("w", delete=True, suffix=".fasta") as tmp1, \
         tempfile.NamedTemporaryFile("w", delete=True, suffix=".fasta") as tmp2:
        tmp1.write(open(path1).read()); tmp1.flush()
        tmp2.write(open(path2).read()); tmp2.flush()
        try:
            result = subprocess.run(
                ["mash", "dist", tmp1.name, tmp2.name],
                capture_output=True, text=True, check=True
            )
            dist = float(result.stdout.strip().split("\t")[2])
        except Exception:
            dist = 1.0
    return (g1, g2), dist


def generate_all_vs_all_pairs(mash_clusters):
    """
    Generate all unique genome pairs within each Mash cluster.
    """
    selected_pairs = []

    for comp in mash_clusters:
        comp = list(comp)
        if len(comp) < 2:
            continue

        for g1, g2 in combinations(sorted(comp), 2):
            selected_pairs.append((g1, g2))

    return selected_pairs


def generate_manifest_all_vs_all_pairs(clusters):
    """
    Generate all unique genome pairs within each Predefined_lineage_cluster.
    clusters: dict mapping lineage -> list of (genome, path, date)
    """
    selected_pairs = []

    for lineage, genomes_info in clusters.items():
        genomes = sorted([g for g, _, _ in genomes_info])
        if len(genomes) < 2:
            continue
        for g1, g2 in combinations(genomes, 2):
            selected_pairs.append((g1, g2))

    return selected_pairs


def prefilter_pairs_by_mash(genome_info, threads, mash_threshold, outdir):
    print("Computing all-vs-all Mash distances in parallel...")
    genomes = [(g, info["path"]) for g, info in genome_info.items() if info["cluster_id"] is not None]
    jobs = [(g1[0], g2[0], g1[1], g2[1]) for g1, g2 in combinations(genomes, 2)]
    
    selected_pairs = []
    mash_table = {}  # store all distances

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(run_mash_distance, job): job for job in jobs}
        for f in as_completed(futures):
            (g1, g2), dist = f.result()
            mash_table[(g1, g2)] = dist
            if dist <= mash_threshold:
                selected_pairs.append((g1, g2))

    print(f"{len(selected_pairs)} genome pairs passed Mash threshold ({mash_threshold}).")
    return selected_pairs, mash_table


def generate_knn_kbo_pairs(mash_clusters, mash_table, k_min):
    """
    Generate sparse kbo comparison pairs using adaptive K-Nearest Neighbors (KNN) within each Mash cluster.

    Parameters
    ----------
    mash_clusters : list of sets
        Connected components from Mash single-linkage clustering.
    mash_table : dict
        {(genome1, genome2): mash_distance}
    k_min : int
        Minimum number of nearest neighbors per genome.

    Returns
    -------
    list of tuple
        Unique undirected genome pairs selected for kbo.
    """

    selected_pairs = set()

    for comp in mash_clusters:
        comp = list(comp)
        cluster_size = len(comp)

        if cluster_size < 2:
            continue

        # --- Adaptive K ---
        k_adaptive = max(k_min, int(math.ceil(math.log(cluster_size))))
        k_effective = min(k_adaptive, cluster_size - 1)

        print(f"Mash cluster size={cluster_size} -> using k={k_effective}")

        for g1 in comp:
            distances = []

            for g2 in comp:
                if g1 == g2:
                    continue

                key = (g1, g2)
                rev_key = (g2, g1)

                if key in mash_table:
                    dist = mash_table[key]
                elif rev_key in mash_table:
                    dist = mash_table[rev_key]
                else:
                    continue

                distances.append((g2, dist))

            # Sort by Mash distance
            distances.sort(key=lambda x: x[1])

            # Select k nearest neighbors
            for g2, _ in distances[:k_effective]:
                pair = tuple(sorted((g1, g2)))
                selected_pairs.add(pair)

    return list(selected_pairs)


def run_kbo_call(ref_path, query_path, kbo_threads):
    try:
        result = subprocess.run(
            [
                "kbo", "call",
                "--reference", str(ref_path),
                str(query_path),
                "--threads", str(kbo_threads)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.splitlines()
    except subprocess.CalledProcessError:
        return None


# kbo wrapper
def process_vcf_and_summarize(
    ref_path,
    query_path,
    outdir,
    cluster_id,
    write_vcf=False,
    kbo_threads=1
):
    ref_tag = Path(ref_path).stem
    query_tag = Path(query_path).stem

    if ref_tag == query_tag:
        return None

    vcf_lines = run_kbo_call(ref_path, query_path, kbo_threads)
    if vcf_lines is None:
        return None

    if write_vcf:
        vcf_file = outdir / f"{ref_tag}_vs_{query_tag}_variants.vcf"
        vcf_file.write_text("\n".join(vcf_lines))

    rows = []
    for line in vcf_lines:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        if fields[7] == ".":
            fields[7] = "SNV"
        rows.append([ref_tag, query_tag, cluster_id] + fields)

    if not rows:
        return None

    return pl.DataFrame(
        rows,
        schema=[
            "ref_genome", "query_genome", "Predefined_lineage_cluster",
            "CHROM", "POS", "ID", "REF", "ALT",
            "QUAL", "FILTER", "INFO", "FORMAT", "unknown"
        ],
        orient="row"
    )


def run_kbo_parallel_pairs(selected_pairs, genome_info, outdir, threads, write_vcf=False):
    print(f"Running kbo comparisons for {len(selected_pairs)} prefiltered pairs across {threads} threads...")
    dfs = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(
                process_vcf_and_summarize,
                genome_info[g1]["path"],
                genome_info[g2]["path"],
                outdir,
                genome_info[g1]["cluster_id"],
                write_vcf,
                1 # KNN uses single-threaded
            )
            for g1, g2 in selected_pairs
        ]

        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                dfs.append(df)

    print("All kbo comparisons complete.")
    if dfs:
        return pl.concat(dfs)

    return pl.DataFrame(
        schema=[
            "ref_genome", "query_genome", "Predefined_lineage_cluster",
            "CHROM", "POS", "ID", "REF", "ALT",
            "QUAL", "FILTER", "INFO", "FORMAT", "unknown"
        ]
    )


# SNV summary, clustering, and writing
def write_summary(enriched_df, outdir, snvs_per_mb, general_snvs_per_year, cluster_rates):
    if enriched_df.is_empty():
        print("Warning: No pairs to write to summary.")
        return

    out_file = outdir / "pairwise_comparison_summary.tsv"

    enriched_df = enriched_df.with_columns([
        pl.lit(snvs_per_mb).alias("config_snvs_per_mb"),
        pl.col("Predefined_lineage_cluster").map_elements(
            lambda x: cluster_rates.get(x, general_snvs_per_year)
        ).alias("config_lineage_specific_snvs_per_year")
    ])

    # Include genome sizes at the end
    enriched_df.select([
        "ref_genome",
        "query_genome",
        "Predefined_lineage_cluster",
        "SNVs",
        "Indels",
        "baseline_SNVs_per_Mb_value",
        "time_weighted_SNV_upper_bound_value",
        "years_apart",
        "ref_genome_isolation_date",
        "query_genome_isolation_date",
        "config_snvs_per_mb",
        "config_lineage_specific_snvs_per_year",
        "ref_genome_size_Mb",
        "query_genome_size_Mb"
    ]).write_csv(out_file, separator="\t")

    print(f"Enriched pairwise summary written to {out_file}")


def summarize_snvs(processed_df, genome_info, genome_sizes_mb, cluster_rates, general_snvs_per_year, snvs_per_mb):
    if processed_df.is_empty():
        return pl.DataFrame()

    summary_df = (
        processed_df
        .group_by(["ref_genome", "query_genome", "Predefined_lineage_cluster"])
        .agg([
            (pl.col("INFO") == "SNV").sum().alias("SNVs"),
            (pl.col("INFO") == "INDEL").sum().alias("Indels")
        ])
    )

    enriched_df = summary_df.with_columns([
        pl.col("ref_genome").replace_strict(genome_sizes_mb).alias("ref_genome_size_Mb"),
        pl.col("query_genome").replace_strict(genome_sizes_mb).alias("query_genome_size_Mb"),
        pl.col("Predefined_lineage_cluster")
          .replace_strict(cluster_rates, default=general_snvs_per_year)
          .alias("snvs_per_year"),
        pl.col("ref_genome")
          .replace_strict({g: i["isolation_date"] for g, i in genome_info.items()})
          .alias("ref_genome_isolation_date"),
        pl.col("query_genome")
          .replace_strict({g: i["isolation_date"] for g, i in genome_info.items()})
          .alias("query_genome_isolation_date"),
    ])

    enriched_df = enriched_df.with_columns([
        ((pl.col("ref_genome_size_Mb") + pl.col("query_genome_size_Mb")) / 2 * snvs_per_mb)
            .alias("baseline_SNVs_per_Mb_value"),
        (
            (
                pl.col("ref_genome_isolation_date")
                  .replace("", None)
                  .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                -
                pl.col("query_genome_isolation_date")
                  .replace("", None)
                  .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            )
            .abs()
            .dt.total_days()
            / 365.25
        )
        .fill_null(0.0)
        .alias("years_apart"),
    ])

    return enriched_df.with_columns([
        (pl.col("baseline_SNVs_per_Mb_value") + pl.col("years_apart") * pl.col("snvs_per_year")
        ).alias("time_weighted_SNV_upper_bound_value")
    ])


def cluster_snvs(
    enriched_df,
    genome_info,
    genome_sizes_mb,
    general_snvs_per_year,
    cluster_rates,
    snvs_per_mb,
    outdir,
    genome_to_cluster=None
):
    """
    Post-hoc SNV clustering within Mash single-linkage clusters + kbo-enriched comparisons. Allows joining of potentially disconnected SNV clusters.
    Assigns sequential time_weighted_SNV_cluster IDs per lineage if genome_to_cluster is not provided.
    Deterministic: larger SNV-connected components get smaller cluster IDs,
    genomes within components sorted alphabetically, singletons after connected components.
    """

    # If genome_to_cluster is provided (from merged SNV clusters), use it
    if genome_to_cluster is None:
        genome_to_cluster = {}

        if enriched_df.is_empty() or "Predefined_lineage_cluster" not in enriched_df.columns:
            # fallback: assign sequential clusters
            out_file = outdir / "time_weighted_SNV_clusters.tsv"
            with open(out_file, "w", newline="") as fh:
                writer = csv.writer(fh, delimiter="\t")
                writer.writerow(["Genome", "Predefined_lineage_cluster", "time_weighted_SNV_cluster"])
                for i, genome in enumerate(sorted(genome_info), start=1):
                    writer.writerow([genome, genome_info[genome]["cluster_id"], i])
            print(f"Time-weighted SNV clusters written to {out_file}")
            return

        all_lineages = enriched_df["Predefined_lineage_cluster"].unique().to_list()

        for lineage in all_lineages:
            lineage_df = enriched_df.filter(pl.col("Predefined_lineage_cluster") == lineage)
            genomes_in_lineage = set(
                lineage_df.select(["ref_genome", "query_genome"])
                .unpivot()
                .unique()
                .get_column("value")
                .to_list()
            )
            for g, info in genome_info.items():
                if info["cluster_id"] == lineage:
                    genomes_in_lineage.add(g)

            if not genomes_in_lineage:
                continue

            # SNV graph
            G = nx.Graph()
            G.add_nodes_from(genomes_in_lineage)

            for row in lineage_df.iter_rows(named=True):
                ref, query = row["ref_genome"], row["query_genome"]
                snvs = row.get("SNVs", 0)
                tw_threshold = row.get("time_weighted_SNV_upper_bound_value", snvs_per_mb)
                if snvs <= tw_threshold:
                    G.add_edge(ref, query)

            # Connected components, largest first
            components = sorted(nx.connected_components(G), key=lambda x: -len(x))
            current_rank = 1
            assigned_genomes = set()
            for comp in components:
                for genome in sorted(comp):
                    genome_to_cluster[genome] = (lineage, current_rank)
                    assigned_genomes.add(genome)
                current_rank += 1

            # Singletons (not connected)
            for genome in sorted(genomes_in_lineage - assigned_genomes):
                genome_to_cluster[genome] = (lineage, current_rank)
                current_rank += 1

        # Any missing genomes
        for genome, info in genome_info.items():
            if genome not in genome_to_cluster and info is not None:
                genome_to_cluster[genome] = (info["cluster_id"], current_rank)
                current_rank += 1

    # Write output: sorted by lineage -> cluster ID -> genome
    out_file = outdir / "time_weighted_SNV_clusters.tsv"
    with open(out_file, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["Genome", "Predefined_lineage_cluster", "time_weighted_SNV_cluster"])
        for genome, (lineage, rank) in sorted(
            genome_to_cluster.items(),
            key=lambda x: (x[1][0], x[1][1], x[0])
        ):
            writer.writerow([genome, lineage, rank])

    print(f"Time-weighted SNV clusters written to {out_file}")

    return genome_to_cluster

def merge_and_update_snv_clusters(
    enriched_df,
    genome_info,
    genome_sizes_mb,
    cluster_rates,
    general_snvs_per_year,
    snv_threshold,
    threads,
    mash_table
):
    """
    Merge SNV clusters using representative genomes, guided by Mash distances.
    Returns updated enriched_df and genome_to_cluster mapping.

    Representative-based merging is constrained to genomes sharing the same Predefined_lineage_cluster.
    If mash_table is empty (e.g., --manifest_all_vs_all), representative merging is skipped.
    """
    print("Starting representative-based SNV cluster merging...")

    if enriched_df.is_empty():
        print("No SNV data available. Skipping merging.")
        return enriched_df, {}

    # Build SNV graph
    G = nx.Graph()
    genomes = set(
        enriched_df.select(["ref_genome", "query_genome"])
        .unpivot()
        .unique()
        .get_column("value")
        .to_list()
    )
    G.add_nodes_from(genomes)

    for row in enriched_df.iter_rows(named=True):
        if row["SNVs"] <= row["time_weighted_SNV_upper_bound_value"]:
            G.add_edge(row["ref_genome"], row["query_genome"])

    components = list(nx.connected_components(G))
    print(f"Initial SNV clusters: {len(components)}")

    # Pick one representative per SNV cluster
    representatives = []
    component_map = {}
    for i, comp in enumerate(components):
        rep = sorted(comp)[0]  # deterministic choice
        representatives.append(rep)
        for g in comp:
            component_map[g] = i

    print(f"{len(representatives)} representatives selected.")

    # Representative-based merging (lineage-aware)
    if mash_table:
        rep_pairs = []
        for rep in representatives:
            rep_lineage = genome_info[rep]["cluster_id"]
            candidate_genomes = [
                g for g in genomes
                if component_map[g] != component_map[rep]
                and genome_info[g]["cluster_id"] == rep_lineage  # lineage check
            ]
            if not candidate_genomes:
                continue
            closest = min(
                candidate_genomes,
                key=lambda g: mash_table.get((rep, g), mash_table.get((g, rep), 1.0))
            )
            rep_pairs.append((rep, closest))

        if rep_pairs:
            print(f"Running KBO on {len(rep_pairs)} cross-cluster representative pairs (lineage-aware)...")
            rep_df = run_kbo_parallel_pairs(rep_pairs, genome_info, Path("."), threads, write_vcf=False)
            if not rep_df.is_empty():
                rep_summary = summarize_snvs(
                    rep_df,
                    genome_info,
                    genome_sizes_mb,
                    cluster_rates,
                    general_snvs_per_year,
                    snv_threshold
                )
                # Append new pairwise comparisons to enriched_df
                enriched_df = pl.concat([enriched_df, rep_summary])

                # Merge SNV clusters if SNVs <= threshold **and same lineage**
                merge_graph = nx.Graph()
                merge_graph.add_nodes_from(range(len(components)))
                for row in rep_summary.iter_rows(named=True):
                    g1, g2, snvs = row["ref_genome"], row["query_genome"], row["SNVs"]
                    lineage1 = genome_info[g1]["cluster_id"]
                    lineage2 = genome_info[g2]["cluster_id"]
                    if snvs <= snv_threshold and lineage1 == lineage2:
                        c1, c2 = component_map[g1], component_map[g2]
                        merge_graph.add_edge(c1, c2)
                merged_components = sorted(
                    nx.connected_components(merge_graph),
                    key=lambda x: -sum(len(components[i]) for i in x)
                )
            else:
                merged_components = [{i} for i in range(len(components))]
        else:
            merged_components = [{i} for i in range(len(components))]
    else:
        # Skip representative merging, just use original SNV components
        merged_components = [comp for comp in components]

    print(f"SNV clusters after merging: {len(merged_components)}")

    # Build final genome -> merged cluster mapping
    genome_to_cluster = {}
    cluster_counter = 1
    for merged_comp in merged_components:
        # merged_comp contains indices of components
        merged_genomes = [g for g, cidx in component_map.items() if cidx in merged_comp]
        for g in sorted(merged_genomes):
            lineage = genome_info[g]["cluster_id"]
            genome_to_cluster[g] = (lineage, cluster_counter)
        cluster_counter += 1

    # Add any genomes missing from SNV graph
    for genome, info in genome_info.items():
        if genome not in genome_to_cluster and info["cluster_id"] is not None:
            lineage = info["cluster_id"]
            genome_to_cluster[genome] = (lineage, cluster_counter)
            cluster_counter += 1

    final_cluster_count = cluster_counter - 1
    print(f"Total SNV clusters (including singletons): {final_cluster_count}")

    return enriched_df, genome_to_cluster


def get_args():
    parser = argparse.ArgumentParser(description="Time-weighted dynamic SNV clustering with kbo + Mash prefilter")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outdir", default="vcf_output_parallel")
    parser.add_argument("--knn_k", type=int, default=10,
                        help="Number of nearest Mash neighbors per genome for kbo comparisons")
    parser.add_argument("--manifest_all_vs_all", action="store_true",
                        help="""Run full all-vs-all kbo within each Predefined_lineage_cluster from manifest file.
                        Skips KNN and MASH pre-filtering. Takes longer to run. User-defined groups to be compared""")
    parser.add_argument("--all_vs_all", action="store_true",
                        help="""Run all-vs-all kbo within each MASH cluster. Ignores Predefined_lineage_cluster from manifest file.
                        Skips KNN. Takes longer to run.""")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--general_snvs_per_year", type=float, default=0.0)
    parser.add_argument("--lineage_specific_snvs_per_year", type=str)
    parser.add_argument("--snvs_per_mb", type=float, default=5.0)
    parser.add_argument("--vcf", action="store_true")
    parser.add_argument("--mash_threshold", type=float, default=0.0008,
                        help="Mash distance threshold for prefiltering genome pairs")
    return parser.parse_args()


def main():
    args = get_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Parse manifest, get genome info and cluster rates
    clusters, genome_info, cluster_rates = parse_manifest_and_rates(
        args.manifest, args.lineage_specific_snvs_per_year
    )
    genome_sizes_mb = get_genome_sizes(genome_info)

    ## Determine kbo pair generation strategy

    mash_table = {}
    mash_clusters = []
    selected_pairs = []

    if args.manifest_all_vs_all:
        print("Using manifest-defined all-vs-all (skipping Mash).")
        selected_pairs = generate_manifest_all_vs_all_pairs(clusters)
        print(f"{len(selected_pairs)} total manifest-cluster genome pairs for kbo.")

    else:
        # Load or compute Mash distances
        mash_outfile = outdir / "all_vs_all_mash_distances.tsv"
        all_mash_pairs = []

        if mash_outfile.exists():
            print(f"Found existing Mash distance table: {mash_outfile}, skipping Mash computation...")
            with open(mash_outfile) as fh:
                next(fh)
                for line in fh:
                    g1, g2, dist = line.strip().split("\t")
                    dist = float(dist)
                    mash_table[(g1, g2)] = dist
                    if dist <= args.mash_threshold:
                        all_mash_pairs.append((g1, g2))
        else:
            all_mash_pairs, mash_table = prefilter_pairs_by_mash(
                genome_info, args.threads, args.mash_threshold, outdir
            )
            with open(mash_outfile, "w") as fh:
                fh.write("genome1\tgenome2\tmash_distance\n")
                for (g1, g2), dist in mash_table.items():
                    fh.write(f"{g1}\t{g2}\t{dist}\n")
            print(f"All-vs-all Mash distances written to {mash_outfile}")

        # Mash single-linkage clusters (lineage-aware)
        G_mash = nx.Graph()
        mash_genomes = [g for g in genome_info if genome_info[g]["cluster_id"] is not None]
        G_mash.add_nodes_from(mash_genomes)
        
        for g1, g2 in all_mash_pairs:
            lineage1 = genome_info[g1]["cluster_id"]
            lineage2 = genome_info[g2]["cluster_id"]
            if lineage1 == lineage2:
                G_mash.add_edge(g1, g2)
        
        mash_clusters = list(nx.connected_components(G_mash))
        print(f"{len(mash_clusters)} Mash single-linkage clusters (lineage-aware) identified")

        # Generate kbo pairs
        if args.all_vs_all:
            print("Mode: Mash all-vs-all")
            selected_pairs = generate_all_vs_all_pairs(mash_clusters)
            print(f"{len(selected_pairs)} total Mash all-vs-all genome pairs for kbo.")
        else:
            print("Mode: adaptive KNN-based selection within Mash clusters")
            selected_pairs = generate_knn_kbo_pairs(
                mash_clusters=mash_clusters,
                mash_table=mash_table,
                k_min=args.knn_k
            )
            print(f"{len(selected_pairs)} total KNN-selected genome pairs for kbo.")

    # Run kbo for all selected pairs
    processed_df = run_kbo_parallel_pairs(
        selected_pairs, genome_info, outdir, args.threads, write_vcf=args.vcf
    )

    # Summarize SNVs (adds SNVs column)
    enriched_df = summarize_snvs(
        processed_df,
        genome_info,
        genome_sizes_mb,
        cluster_rates,
        args.general_snvs_per_year,
        args.snvs_per_mb
    )

    # Merge clusters using representatives + Mash guidance
    if args.manifest_all_vs_all:
        print("Skipping representative merging, but performing SNV graph clustering.")
        genome_to_cluster = None

    else:
        print("Running representative-based SNV cluster merging...")
        enriched_df, genome_to_cluster = merge_and_update_snv_clusters(
            enriched_df,
            genome_info,
            genome_sizes_mb,
            cluster_rates,
            args.general_snvs_per_year,
            snv_threshold=args.snvs_per_mb,
            threads=args.threads,
            mash_table=mash_table
        )
    
    # Write enriched pairwise summary
    write_summary(
        enriched_df,
        outdir,
        args.snvs_per_mb,
        args.general_snvs_per_year,
        cluster_rates
    )
    
    # Pass updated clusters to cluster_snvs
    cluster_snvs(
        enriched_df,
        genome_info,
        genome_sizes_mb,
        args.general_snvs_per_year,
        cluster_rates,
        args.snvs_per_mb,
        outdir,
        genome_to_cluster=genome_to_cluster
    )


if __name__ == "__main__":
    main()
