# Name-Surname: UMUT YILDIRIM
# ID: 922797

import time

import pandas as pd
import pg8000
from neo4j import GraphDatabase
from faker import Faker
import random
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import col
import matplotlib.pyplot as plt


# Class to run and measure performance of relational vs graph model operations
class GraphQueryRunner:
    def __init__(self, pg_config, neo4j_config):
        # Initialize PostgreSQL connection
        self.pg_conn = pg8000.connect(**pg_config)
        self.pg_cur = self.pg_conn.cursor()

        # Initialize Neo4j connection with Bolt protocol
        uri = f"bolt://{neo4j_config['host']}:{neo4j_config['port']}"
        auth = (neo4j_config['user'], neo4j_config['password'])
        self.neo_drv = GraphDatabase.driver(uri, auth=auth)

        # Faker instance for synthetic data generation
        self.fake = Faker()

        # Initialize Spark session including GraphFrames and JDBC packages
        self.spark = (
            SparkSession.builder
            .appName("Part2Performance")
            .master("local[*]")
            .config(
                "spark.jars.packages",
                ",".join([
                    "graphframes:graphframes:0.8.2-spark3.1-s_2.12",
                    "org.postgresql:postgresql:42.6.2",
                    "org.scala-lang:scala-library:2.12.15"
                ])
            )
            .getOrCreate()
        )

    def clear_data(self):
        """Reset both relational and graph databases to empty state."""
        # Truncate PostgreSQL tables and reset their identities
        self.pg_cur.execute("TRUNCATE TABLE follows, users RESTART IDENTITY CASCADE;")
        self.pg_conn.commit()

        # Delete all nodes and relationships in Neo4j
        with self.neo_drv.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def generate_data(self, num_users, conn_pct):
        """Populate both databases with synthetic users and random follow edges."""
        user_ids = []
        # Create users in PostgreSQL and Neo4j
        for _ in range(num_users):
            name = self.fake.first_name()
            # Insert user into PostgreSQL
            self.pg_cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id", (name,))
            uid = self.pg_cur.fetchone()[0]
            user_ids.append(uid)
            # Create corresponding node in Neo4j
            with self.neo_drv.session() as session:
                session.run("CREATE (u:User {id:$id, name:$name})", id=uid, name=name)
        self.pg_conn.commit()

        # Generate all possible directed edges excluding self-follows
        possible = [(u, v) for u in user_ids for v in user_ids if u != v]
        # Sample a percentage of possible edges
        sample_count = int(len(possible) * conn_pct)
        edges = random.sample(possible, sample_count)

        # Insert follow relationships into both systems
        for src, dst in edges:
            # Relational: follows table
            self.pg_cur.execute("INSERT INTO follows (source_id,target_id) VALUES (%s,%s)", (src, dst))
            # Graph: create a FOLLOWS relationship in Neo4j
            with self.neo_drv.session() as session:
                session.run(
                    "MATCH (a:User {id:$src}), (b:User {id:$dst}) CREATE (a)-[:FOLLOWS]->(b)",
                    src=src, dst=dst
                )
        self.pg_conn.commit()

    def time_cypher(self, query, params):
        """Measure execution time of a Neo4j Cypher query."""
        start = time.time()
        with self.neo_drv.session() as session:
            list(session.run(query, params))  # force consumption
        return time.time() - start

    def time_gf(self, df):
        """Measure execution time of a GraphFrame query result collection."""
        start = time.time()
        df.collect()
        return time.time() - start

    def time_rdd(self, rdd):
        """Measure execution time of an RDD query result collection."""
        start = time.time()
        rdd.collect()
        return time.time() - start

    # Exercise 1: Neo4j Cypher for transitive follows
    def ex1_cypher(self, user_id):
        query = (
            "MATCH (u:User {id:$uid})-[:FOLLOWS*1..]->(v) "
            "RETURN DISTINCT v.id AS id"
        )
        return self.time_cypher(query, {'uid': user_id})

    # Exercise 2: GraphFrames BFS using DataFrames API
    def ex2_graphframes(self, user_id):
        """Use GraphFrames BFS to get all reachable nodes from a start user."""
        # Load relational tables into DataFrames via JDBC
        users_df = (
            self.spark.read.format("jdbc")
            .options(url=self.jdbc_url, dbtable="users", **self.jdbc_props)
            .load()
        )
        follows_df = (
            self.spark.read.format("jdbc")
            .options(url=self.jdbc_url, dbtable="follows", **self.jdbc_props)
            .load()
        )
        # Prepare vertices and edges for GraphFrame
        vertices = users_df.select(col("id").alias("id"))
        edges = follows_df.select(col("source_id").alias("src"), col("target_id").alias("dst"))
        gf = GraphFrame(vertices, edges)
        # BFS without max depth limit to find all reachable
        paths = gf.bfs(fromExpr=f"id = {user_id}", toExpr="true")
        return self.time_gf(paths)

    # Exercise 3: Pure RDD BFS with allowed operations
    def ex3_rdd(self, user_id):
        """Iterative BFS using only basic RDD transformations and actions."""
        # Load follows relations as RDD of (src, dst)
        follows_rdd = (
            self.spark.read.format("jdbc")
            .options(url=self.jdbc_url, dbtable="follows", **self.jdbc_props)
            .load()
            .rdd.map(lambda r: (r['source_id'], r['target_id']))
        )

        start_time = time.time()
        # Initialize RDDs for BFS levels
        sc = self.spark.sparkContext
        current = sc.parallelize([user_id])  # start from given user
        reached = current

        while True:
            # Join current frontier with follows edges
            next_level = (
                current.cartesian(follows_rdd)
                .filter(lambda x: x[0] == x[1][0])  # match src
                .map(lambda x: x[1][1])  # extract dst
                .distinct()
            )
            # Exclude already reached nodes
            new_nodes = next_level.subtract(reached)
            if new_nodes.isEmpty():
                break
            # Update frontier and reached set
            reached = reached.union(new_nodes)
            current = new_nodes

        # Exclude the start node and collect final results
        reached.filter(lambda x: x != user_id).collect()
        return time.time() - start_time


if __name__ == "__main__":
    # Database connection configurations
    pg_cfg = {'user': 'postgres', 'password': '15022001', 'database': 'mydb', 'host': 'localhost', 'port': 5432}
    neo_cfg = {'host': 'localhost', 'port': 7687, 'user': 'neo4j', 'password': '15022001'}

    # Initialize runner and set JDBC props for Spark
    runner = GraphQueryRunner(pg_cfg, neo_cfg)
    runner.jdbc_url = f"jdbc:postgresql://{pg_cfg['host']}:{pg_cfg['port']}/{pg_cfg['database']}"
    runner.jdbc_props = {"user": pg_cfg['user'], "password": pg_cfg['password'], "driver": "org.postgresql.Driver"}

    # Define test scenarios: (number_of_users, connection_percentage)
    scenarios = [(n, p) for n in [10, 100, 1000] for p in [0.05, 0.10, 0.25]]
    results = {}

    # Run each scenario and record timings
    for num_users, pct in scenarios:
        label = f"U{num_users}_{int(pct * 100)}pct"
        print(f"=== Scenario {label} ===")
        runner.clear_data()
        runner.generate_data(num_users, pct)
        uid = 1  # test from user with id 1

        t1 = runner.ex1_cypher(uid)
        t2 = runner.ex2_graphframes(uid)
        t3 = runner.ex3_rdd(uid)

        results[label] = {'Cypher': t1, 'GraphFrames': t2, 'RDD': t3}
        print(f"--> Part2 Exercise1 - Cypher    : {t1:.6f}s")
        print(f"--> Part2 Exercise2 - GraphFrames: {t2:.6f}s")
        print(f"--> Part2 Exercise3 - RDD only  : {t3:.6f}s\n")


    def plot_performance(data, normalize=False):
        """Generate grouped bar charts comparing execution times."""
        labels = list(data.keys())
        methods = list(next(iter(data.values())).keys())

        matrix = {m: [data[s][m] for s in labels] for m in methods}
        if normalize:
            for m in methods:
                vals = matrix[m]
                mn, mx = min(vals), max(vals)
                matrix[m] = [(v - mn) / (mx - mn) if mx > mn else 0 for v in vals]

        x = range(len(labels))
        width = 0.8 / len(methods)
        fig, ax = plt.subplots()
        for i, m in enumerate(methods):
            ax.bar([xi + i * width for xi in x], matrix[m], width=width, label=m)

        ax.set_xticks([xi + width * (len(methods) / 2 - 0.5) for xi in x])
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Normalized Time' if normalize else 'Time (s)')
        ax.set_title('Graph vs Relational Performance' + (' (Normalized)' if normalize else ''))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


    def save_graph_results(results_dict, filename="performance_results_part2.xlsx"):
        data = []
        for scenario, times in results_dict.items():
            parts = scenario.split('_')
            users = int(parts[0][1:])
            conn_rate = int(parts[1][:-3]) / 100
            row = {
                'Scenario': scenario,
                'Users': users,
                'Connection Rate': conn_rate,
                'Cypher (s)': times['Cypher'],
                'GraphFrames (s)': times['GraphFrames'],
                'RDD (s)': times['RDD']
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)


    save_graph_results(results)
    # Plot raw and normalized charts
    plot_performance(results, normalize=False)
    plot_performance(results, normalize=True)
