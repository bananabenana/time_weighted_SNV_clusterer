#!/usr/bin/env python3
import argparse
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


# KBO wrapper
def process_vcf_and_summarize(ref_path, query_path, outdir, cluster_id, write_vcf=False):
    ref_tag = Path(ref_path).stem
    query_tag = Path(query_path).stem
    if ref_tag == query_tag:
        return None

    try:
        result = subprocess.run(
            ["kbo", "call", "--reference", str(ref_path), str(query_path), "--threads", "1"],
            capture_output=True, text=True, check=True
        )
        vcf_lines = result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error running kbo for {ref_tag} vs {query_tag}: {e}")
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
                "prefiltered",
                write_vcf
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


def cluster_snvs(enriched_df, genome_info, genome_sizes_mb, general_snvs_per_year, cluster_rates, snvs_per_mb, outdir):
    """
    Post-hoc SNV clustering within Mash single-linkage clusters.
    All genomes, including singletons, are assigned sequential
    time_weighted_SNV_cluster IDs per lineage.
    The Predefined_lineage_cluster column uses the input manifest values.
    Output is sorted by ascending time_weighted_SNV_cluster within each lineage.
    """
    genome_to_cluster = {}
    # Fallback: no SNV-enriched data (e.g. no Mash-linked genomes)
    if (
        enriched_df.is_empty()
        or "Predefined_lineage_cluster" not in enriched_df.columns
    ):
        out_file = outdir / "time_weighted_SNV_clusters.tsv"
    
        with open(out_file, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow([
                "Genome",
                "Predefined_lineage_cluster",
                "time_weighted_SNV_cluster",
            ])
    
            for i, genome in enumerate(sorted(genome_info), start=1):
                writer.writerow([
                    genome,
                    genome_info[genome]["cluster_id"],
                    i,
                ])
    
        return
    
    # Process each predefined lineage separately
    for lineage in enriched_df["Predefined_lineage_cluster"].unique().to_list():
        lineage_df = enriched_df.filter(pl.col("Predefined_lineage_cluster") == lineage)

        # Collect all genomes in this lineage (from genome_info + enriched_df)
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

        # Build SNV-based graph
        G = nx.Graph()
        G.add_nodes_from(genomes_in_lineage)

        for row in lineage_df.iter_rows(named=True):
            ref, query, snvs, tw_threshold = (
                row["ref_genome"],
                row["query_genome"],
                row["SNVs"],
                row["time_weighted_SNV_upper_bound_value"]
            )
            if snvs <= tw_threshold:
                G.add_edge(ref, query)

        # Connected components → SNV clusters
        components = list(nx.connected_components(G))
        components = sorted(components, key=lambda x: -len(x))  # largest first

        # Assign sequential cluster IDs for connected components
        current_rank = 1
        assigned_genomes = set()
        for comp in components:
            for genome in sorted(comp):  # sort inside component for reproducibility
                genome_to_cluster[genome] = (lineage, current_rank)
                assigned_genomes.add(genome)
            current_rank += 1

        # Add any unconnected genomes (singletons not in enriched_df)
        remaining = genomes_in_lineage - assigned_genomes
        for genome in sorted(remaining):  # sorted for reproducibility
            genome_to_cluster[genome] = (lineage, current_rank)
            current_rank += 1

    # Add any genomes completely missing from enriched_df (should rarely happen)
    for genome, info in genome_info.items():
        if genome not in genome_to_cluster and info["cluster_id"] is not None:
            genome_to_cluster[genome] = (info["cluster_id"], current_rank)
            current_rank += 1

    # Write output sorted by lineage then ascending cluster ID
    out_file = outdir / "time_weighted_SNV_clusters.tsv"
    with open(out_file, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["Genome", "Predefined_lineage_cluster", "time_weighted_SNV_cluster"])
        for genome, (lineage, rank) in sorted(
            genome_to_cluster.items(), key=lambda x: (x[1][1], x[1][0], x[0])
        ):
            # Use the original lineage from genome_info
            writer.writerow([genome, genome_info[genome]["cluster_id"], rank])

    print(f"Time-weighted SNV clusters written to {out_file}")


def get_args():
    parser = argparse.ArgumentParser(description="Time-weighted dynamic SNV clustering with kbo + Mash prefilter")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outdir", default="vcf_output_parallel")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--general_snvs_per_year", type=float, default=0.0)
    parser.add_argument("--lineage_specific_snvs_per_year", type=str)
    parser.add_argument("--snvs_per_mb", type=float, default=5.0)
    parser.add_argument("--vcf", action="store_true")
    parser.add_argument("--mash_threshold", type=float, default=1e-5, help="Mash distance threshold for prefiltering genome pairs")
    return parser.parse_args()


def main():
    args = get_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Parse manifest and optional SNV rates
    clusters, genome_info, cluster_rates = parse_manifest_and_rates(
        args.manifest, args.lineage_specific_snvs_per_year
    )
    genome_sizes_mb = get_genome_sizes(genome_info)

    # Mash prefilter to prevent all vs all kbo...
    all_mash_pairs, mash_table = prefilter_pairs_by_mash(
        genome_info, args.threads, args.mash_threshold, outdir
    )

    # Write Mash all-vs-all distance table
    mash_outfile = outdir / "all_vs_all_mash_distances.tsv"
    with open(mash_outfile, "w") as fh:
        fh.write("genome1\tgenome2\tmash_distance\n")
        for (g1, g2), dist in mash_table.items():
            fh.write(f"{g1}\t{g2}\t{dist}\n")
    print(f"All-vs-all Mash distances written to {mash_outfile}")

    # Build Mash single-linkage clusters
    G_mash = nx.Graph()
    mash_genomes = [g for g in genome_info if genome_info[g]["cluster_id"] is not None]
    G_mash.add_nodes_from(mash_genomes)
    G_mash.add_edges_from(all_mash_pairs)
    mash_clusters = list(nx.connected_components(G_mash))
    print(f"{len(mash_clusters)} Mash single-linkage clusters identified")

    # Generate ALL KBO pairs within each Mash cluster (no early stop)
    selected_pairs = []
    for comp in mash_clusters:
        comp = list(comp)
        if len(comp) < 2:
            continue
        # All-vs-all combinations
        for i, g1 in enumerate(comp):
            for g2 in comp[i+1:]:
                selected_pairs.append((g1, g2))

    # Run KBO for all pairs
    processed_df = run_kbo_parallel_pairs(
        selected_pairs, genome_info, outdir, args.threads, write_vcf=args.vcf
    )

    # Summarize SNVs
    enriched_df = summarize_snvs(
        processed_df,
        genome_info,
        genome_sizes_mb,
        cluster_rates,
        args.general_snvs_per_year,
        args.snvs_per_mb
    )

    # Write pairwise summary
    write_summary(
        enriched_df,
        outdir,
        args.snvs_per_mb,
        args.general_snvs_per_year,
        cluster_rates
    )

    # Post-hoc SNV clustering + assign sequential cluster IDs per lineage, handling singletons as unique SNV clusters
    cluster_snvs(
        enriched_df,
        genome_info,
        genome_sizes_mb,
        args.general_snvs_per_year,
        cluster_rates,
        args.snvs_per_mb,
        outdir
    )


if __name__ == "__main__":
    main()
