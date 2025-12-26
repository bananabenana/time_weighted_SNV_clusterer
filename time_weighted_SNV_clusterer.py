#!/usr/bin/env python3
import argparse
import subprocess
from itertools import combinations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import csv
import networkx as nx
import polars as pl


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


def process_vcf_and_summarize(ref_path, query_path, outdir, cluster_id, write_vcf=False):
    ref_tag = Path(ref_path).stem
    query_tag = Path(query_path).stem
    if ref_tag == query_tag:
        return None

    print(f"Running kbo for {ref_tag} vs {query_tag}")
    try:
        result = subprocess.run(
            ["kbo", "call", "--reference", str(ref_path), str(query_path), "--threads", "1"],
            capture_output=True,
            text=True,
            check=True
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


def run_kbo_parallel(clusters, outdir, threads, write_vcf=False):
    print(f"Running kbo comparisons in parallel across {threads} threads...")
    dfs = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(
                process_vcf_and_summarize,
                ref_path,
                query_path,
                outdir,
                cluster_id,
                write_vcf
            )
            for cluster_id, genomes in clusters.items()
            for (_, ref_path, _), (_, query_path, _) in combinations(genomes, 2)
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


def write_summary(enriched_df, outdir, snvs_per_mb, general_snvs_per_year, cluster_rates):
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
        (((pl.col("ref_genome_isolation_date").str.strptime(pl.Date, "%Y-%m-%d") -
           pl.col("query_genome_isolation_date").str.strptime(pl.Date, "%Y-%m-%d")).abs()
          ).cast(pl.Int64) / 1e9 / 86400 / 365.25).alias("years_apart"),
    ])

    return enriched_df.with_columns([
        (pl.col("baseline_SNVs_per_Mb_value") + pl.col("years_apart") * pl.col("snvs_per_year")
        ).alias("time_weighted_SNV_upper_bound_value")
    ])


def cluster_snvs(enriched_df, genome_info, genome_sizes_mb, general_snvs_per_year, cluster_rates, snvs_per_mb, outdir):
    results = []
    global_cluster_counter = 1

    cluster_to_genomes = defaultdict(list)
    for genome, info in genome_info.items():
        if info["cluster_id"] is not None:
            cluster_to_genomes[info["cluster_id"]].append(genome)

    for cluster_id, genomes in cluster_to_genomes.items():
        G = nx.Graph()
        G.add_nodes_from(genomes)

        cluster_rows = enriched_df.filter(pl.col("Predefined_lineage_cluster") == cluster_id)
        for row in cluster_rows.iter_rows(named=True):
            ref, query, snvs = row["ref_genome"], row["query_genome"], row["SNVs"]
            ref_mb, query_mb = genome_sizes_mb[ref], genome_sizes_mb[query]
            years = row["years_apart"]
            rate = cluster_rates.get(cluster_id, general_snvs_per_year)

            threshold = snvs_per_mb * ((ref_mb + query_mb) / 2) + years * rate
            if snvs <= threshold:
                G.add_edge(ref, query)

        for comp in nx.connected_components(G):
            for genome in comp:
                results.append((genome, cluster_id, global_cluster_counter))
            global_cluster_counter += 1

    for genome, info in genome_info.items():
        if info["cluster_id"] is None:
            results.append((genome, "", "NOT ASSIGNED DUE TO MISSING Predefined_lineage_cluster"))

    out_file = outdir / "time_weighted_SNV_clusters.tsv"
    with open(out_file, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["Genome", "Predefined_lineage_cluster", "time_weighted_SNV_cluster"])
        writer.writerows(results)

    print(f"Time-weighted SNV clusters written to {out_file}")


def get_args():
    parser = argparse.ArgumentParser(description="Time-weighted dynamic SNV clustering with kbo + Polars")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outdir", default="vcf_output_parallel")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--general_snvs_per_year", type=float, required=True)
    parser.add_argument("--lineage_specific_snvs_per_year", type=str)
    parser.add_argument("--snvs_per_mb", type=float, default=5.0)
    parser.add_argument("--vcf", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    clusters, genome_info, cluster_rates = parse_manifest_and_rates(
        args.manifest, args.lineage_specific_snvs_per_year
    )

    clusters = {k: v for k, v in clusters.items() if len(v) > 1}

    genome_sizes_mb = get_genome_sizes(genome_info)

    processed_df = run_kbo_parallel(
        clusters, outdir, args.threads, write_vcf=args.vcf
    )

    enriched_df = summarize_snvs(
        processed_df,
        genome_info,
        genome_sizes_mb,
        cluster_rates,
        args.general_snvs_per_year,
        args.snvs_per_mb
    )

    # Save summary dataframe
    write_summary(enriched_df, outdir, args.snvs_per_mb, args.general_snvs_per_year, cluster_rates)

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
