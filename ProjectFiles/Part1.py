# Name-Surname: UMUT YILDIRIM
# ID: 922797

import time
import pandas as pd
import pg8000
from matplotlib import pyplot as plt
from pymongo import MongoClient
from datetime import datetime, time as dt_time
from dateutil.relativedelta import relativedelta
from faker import Faker
import random
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, explode_outer, col


def process_inner_join(grouped_data):
    """Process RDD grouped data for an inner join: keep only keys with both employees and company info."""
    key, values = grouped_data
    employees = None
    company = None
    for source, payload in values:
        if source == 'employees':
            employees = payload
        elif source == 'company':
            company = payload
    return key, {'company': company, 'employees': employees}

def process_company_join(grouped_data):
    """Process grouped data for left join from companies to persons."""
    key, values = grouped_data
    company = None
    employees = []
    for source, payload in values:
        if source == 'company':
            company = payload
        elif source == 'employee':
            employees.append(payload)
    return key, {'company': company, 'employees': employees}

# Class to run and measure performance of relational vs document model operations
class QueryRunner:
    def __init__(self, pg_config, mongo_config):
        # Establish PostgreSQL connection
        self.pg_conn = pg8000.connect(**pg_config)
        self.pg_cur = self.pg_conn.cursor()

        # Establish MongoDB connection
        self.mongo_client = MongoClient(**mongo_config)
        self.mongo_db = self.mongo_client[pg_config['database']]
        # URI used by Spark's MongoDB connector
        self.mongo_uri = f"mongodb://{mongo_config['host']}:{mongo_config['port']}/{self.mongo_db.name}"

        # Faker instance for synthetic data generation
        self.fake = Faker()

        # Initialize Spark session with MongoDB connector
        self.spark = (
            SparkSession.builder
            .appName("PerformanceTest")
            .master("local[*]")
            .config("spark.jars.packages",
                    "org.mongodb.spark:mongo-spark-connector_2.12:10.5.0")
            .config("spark.mongodb.read.connection.uri", self.mongo_uri)
            .config("spark.mongodb.read.database", self.mongo_db.name)
            .getOrCreate()
        )

        # JDBC settings for PostgreSQL via Spark if needed
        self.jdbc_url = f"jdbc:postgresql://{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
        self.jdbc_properties = {
            "user": pg_config['user'],
            "password": pg_config['password'],
            "driver": "org.postgresql.Driver"
        }

        # Predefined SQL queries for Exercises 1-4
        self.sql_queries = {
            "--> Part1 Exercise 1":  # inner join: only employed persons
                """
                SELECT p.id, p.name, p.surname,
                       c.id AS company_id, c.name AS company_name
                FROM person p
                JOIN works_in w ON p.id = w.person_id
                JOIN company c ON w.company_id = c.id;
                """,
            "--> Part1 Exercise 2":  # left join from persons: persons + their company if any
                """
                SELECT p.id, p.name, p.surname,
                       c.id AS company_id, c.name AS company_name
                FROM person p
                LEFT JOIN works_in w ON p.id = w.person_id
                LEFT JOIN company c ON w.company_id = c.id;
                """,
            "--> Part1 Exercise 3":  # left join from companies: companies + employed persons
                """
                SELECT c.id, c.name,
                       p.id AS person_id, p.name AS person_name, p.surname AS person_surname
                FROM company c
                LEFT JOIN works_in w ON c.id = w.company_id
                LEFT JOIN person p ON w.person_id = p.id;
                """,
            "--> Part1 Exercise 4":  # full outer join: all persons & companies, matching employed ones
                """
                SELECT p.id AS person_id, p.name AS person_name,
                       c.id AS company_id, c.name AS company_name
                FROM works_in w
                FULL JOIN person p ON w.person_id = p.id
                FULL JOIN company c ON w.company_id = c.id;
                """
        }

    def clear_tables(self):
        """Truncate all relational tables and drop MongoDB collections to reset state."""
        self.pg_cur.execute(
            "TRUNCATE TABLE works_in, person, company RESTART IDENTITY CASCADE;"
        )
        self.pg_conn.commit()
        self.mongo_db.persons.drop()
        self.mongo_db.companies.drop()

    def generate_data(self, num_persons, num_companies, occupancy_rate):
        """Generate data and insert into both PostgreSQL and MongoDB.
        - occupancy_rate: fraction of persons employed (%10 %25 %50)"""
        sectors = ['automotive', 'banking', 'services', 'healthcare', 'chemicals', 'public']
        company_ids = []
        # Create companies
        for _ in range(num_companies):
            name = self.fake.company()
            fd_date = self.fake.date_between(start_date='-30y', end_date='today')
            sector = random.choice(sectors)
            # Insert into PostgreSQL
            self.pg_cur.execute(
                "INSERT INTO company (name, founding_date, sector) VALUES (%s,%s,%s) RETURNING id",
                (name, fd_date, sector)
            )
            cid = self.pg_cur.fetchone()[0]
            company_ids.append(cid)
            # Mirror into MongoDB
            self.mongo_db.companies.insert_one({
                "_id": cid,
                "name": name,
                "founding_date": datetime.combine(fd_date, dt_time.min),
                "sector": sector
            })
        self.pg_conn.commit()

        # Create persons and optional employments
        for _ in range(num_persons):
            first, last = self.fake.first_name(), self.fake.last_name()
            bd = self.fake.date_between(start_date='-80y', end_date='-18y')
            addr, country = self.fake.address(), self.fake.country()
            self.pg_cur.execute(
                "INSERT INTO person (name,surname,birthdate,residence_address,country)"
                " VALUES (%s,%s,%s,%s,%s) RETURNING id",
                (first, last, bd, addr, country)
            )
            pid = self.pg_cur.fetchone()[0]
            doc = {
                "_id": pid, "name": first, "surname": last,
                "birthdate": datetime.combine(bd, dt_time.min),
                "residence_address": addr, "country": country,
                "employments": []
            }
            # Randomly assign employment based on occupancy_rate
            if company_ids and random.random() < occupancy_rate:
                comp = random.choice(company_ids)
                min_start = bd + relativedelta(years=18)
                st = self.fake.date_between(start_date=min_start, end_date='today')
                self.pg_cur.execute(
                    "INSERT INTO works_in (person_id,company_id,start_date) VALUES (%s,%s,%s)",
                    (pid, comp, st)
                )
                doc['employments'].append({
                    "company_id": comp,
                    "start_date": datetime.combine(st, dt_time.min),
                    "end_date": None
                })
            self.mongo_db.persons.insert_one(doc)
        self.pg_conn.commit()

    def load_spark(self):
        """Load MongoDB collections into Spark DataFrames and RDDs for later queries."""
        self.persons = (
            self.spark.read
                .format("mongodb")
                .option("collection", "persons")
                .load()
        )
        self.companies = (
            self.spark.read
                .format("mongodb")
                .option("collection", "companies")
                .load()
        )
        # Convert DataFrames to RDDs keyed by document _id
        self.rdd_person = self.persons.rdd.map(lambda r: (r['_id'], r['employments']))
        self.rdd_company = self.companies.rdd.map(lambda r: (r['_id'], r))

    # Timing wrappers for different query types
    def time_sql(self, sql):
        start = time.time()
        self.pg_cur.execute(sql)
        self.pg_cur.fetchall()
        return time.time() - start

    def time_df(self, df):
        start = time.time()
        df.collect()
        return time.time() - start

    def time_rdd(self, rdd):
        start = time.time()
        rdd.collect()
        return time.time() - start

    # Spark DataFrame implementations for Exercises 5-8
    def df5(self):
        """Exercise 5: Inner join"""
        exploded = self.persons.withColumn("emp", explode(col("employments")))

        # Inner join with company informations
        result = exploded.join(
            self.companies,
            exploded.emp.company_id == self.companies._id,
            how='inner'
        ).select(
            self.persons._id.alias("person_id"),
            self.persons.name.alias("person_name"),
            self.persons.surname.alias("person_surname"),
            self.companies._id.alias("company_id"),
            self.companies.name.alias("company_name"),
            self.companies.sector,
            exploded.emp.start_date
        )

        return result

    def df6(self):
        """Exercise 6: All persons and companies they work if any"""
        # Tüm kişileri al, employment olmasa bile (explode_outer)
        exploded = self.persons.withColumn("emp", explode_outer(col("employments")))

        result = exploded.join(
            self.companies,
            exploded.emp.company_id == self.companies._id,
            how='left'
        ).select(
            self.persons._id.alias("person_id"),
            self.persons.name.alias("person_name"),
            self.persons.surname.alias("person_surname"),
            self.companies._id.alias("company_id"),
            self.companies.name.alias("company_name"),
            self.companies.sector,
            exploded.emp.start_date
        )

        return result

    def df7(self):
        """Exercise 7: All companies and employees if any"""
        exploded = self.persons.withColumn("emp", explode_outer(col("employments")))

        # Left join
        result = self.companies.join(
            exploded,
            self.companies._id == exploded.emp.company_id,
            how='left'
        ).select(
            self.companies._id.alias("company_id"),
            self.companies.name.alias("company_name"),
            self.companies.sector,
            self.companies.founding_date,
            exploded._id.alias("person_id"),
            exploded.name.alias("person_name"),
            exploded.surname.alias("person_surname"),
            exploded.emp.start_date
        )

        return result

    def df8(self):
        """Exercise 8: Full outer join"""
        exploded = self.persons.withColumn("emp", explode_outer(col("employments")))

        # Full outer join
        result = exploded.join(
            self.companies,
            exploded.emp.company_id == self.companies._id,
            how='full'
        ).select(
            exploded._id.alias("person_id"),
            exploded.name.alias("person_name"),
            exploded.surname.alias("person_surname"),
            self.companies._id.alias("company_id"),
            self.companies.name.alias("company_name"),
            self.companies.sector,
            exploded.emp.start_date
        )

        return result

    # RDD implementations for Exercises 9-12 using only basic RDD ops
    def rdd9(self):
        """Exercise 9: inner join via RDD primitives"""
        # Map person to pairs of (company_id, person_id) for employed persons
        employed_pairs = self.rdd_person.flatMap(
            lambda x: [(employment['company_id'], x[0]) for employment in x[1]] if x[1] else []
        )
        # Group employees by company and join with company data
        company_employees = employed_pairs.groupByKey().map(lambda x: (x[0], list(x[1])))
        company_lookup = self.rdd_company.map(lambda x: (x[0], x[1]))
        unioned = (
            company_employees.map(lambda x: (x[0], ('employees', x[1])))
            .union(company_lookup.map(lambda x: (x[0], ('company', x[1]))))
        )
        # Filter keys present in both sets and process join
        return unioned.groupByKey() \
                      .filter(lambda x: len(list(x[1])) == 2) \
                      .map(process_inner_join)

    def rdd10(self):
        """Exercise 10: returns persons with/without companies (RDD only)
        """
        person_data_for_rdd10 = self.persons.rdd.map(
            lambda r: (r['_id'], (r['name'], r['surname'],
                                  [emp.company_id for emp in r['employments']] if r['employments'] else []))
        )

        company_data_for_rdd10 = self.companies.rdd.map(
            lambda r: (r['_id'], (r['name'], r['sector']))
        )

        persons_to_join_with_comp = person_data_for_rdd10.filter(lambda x: x[1][2]) \
            .flatMap(lambda x: [(cid, ('p_info', (x[0], x[1][0], x[1][1]))) for cid in x[1][2]])

        companies_to_be_joined = company_data_for_rdd10.map(lambda x: (x[0], ('c_info', x[1])))

        # Union and groupByKey to simulate left join logic for persons to companies
        grouped_by_cid = persons_to_join_with_comp.union(companies_to_be_joined).groupByKey()

        def process_rdd10_group(item):
            cid, data_iterable = item
            company_details = None
            persons_linked_to_this_cid = []

            for tag, payload in data_iterable:
                if tag == 'c_info':
                    company_details = payload
                elif tag == 'p_info':
                    persons_linked_to_this_cid.append(payload)

            output = []
            if company_details:  # Company info must exist for the join
                for p_info in persons_linked_to_this_cid:
                    output.append((p_info[0], {
                        'person_name': p_info[1],
                        'person_surname': p_info[2],
                        'company_id': cid,
                        'company_name': company_details[0],
                        'company_sector': company_details[1]
                    }))
            return output

        persons_with_company_details = grouped_by_cid.flatMap(process_rdd10_group)

        # Persons without any company employments
        persons_without_company_details = person_data_for_rdd10.filter(lambda x: not x[1][2]) \
            .map(lambda x: (x[0], {
            'person_name': x[1][0],
            'person_surname': x[1][1],
            'company_id': None,
            'company_name': None,
            'company_sector': None
        }))

        return persons_with_company_details.union(persons_without_company_details)

    def rdd11(self):
        """Exercise 11: left join companies->persons"""
        company_keys = self.rdd_company.map(lambda x: (x[0], x[1]))
        emp_pairs = self.rdd_person.flatMap(
            lambda x: [(employment['company_id'], x[0]) for employment in x[1]] if x[1] else []
        )
        unioned = (
            company_keys.map(lambda x: (x[0], ('company', x[1])))
            .union(emp_pairs.map(lambda x: (x[0], ('employee', x[1]))))
        )
        return unioned.groupByKey().map(process_company_join)

    def rdd12(self):
        """Exercise 12: full join implementation (RDD only)
        """
        persons_full_rdd = self.persons.rdd.map(lambda r: (r['_id'], r))
        companies_full_rdd = self.companies.rdd.map(lambda r: (r['_id'], r))

        # Matched person-company pairs with details
        cid_with_person_details = persons_full_rdd.flatMap(
            lambda p_row_tuple: [
                (emp.company_id,
                 ('person_match_info', (p_row_tuple[0], p_row_tuple[1]['name'], p_row_tuple[1]['surname'])))
                for emp in p_row_tuple[1]['employments']
            ] if p_row_tuple[1]['employments'] else []
        )

        cid_with_company_details = companies_full_rdd.map(
            lambda c_row_tuple: (
            c_row_tuple[0], ('company_match_info', (c_row_tuple[1]['name'], c_row_tuple[1]['sector'])))
        )

        # Union and group by cid to find matches
        grouped_for_match = cid_with_person_details.union(cid_with_company_details).groupByKey()

        def process_rdd12_match_group(item):
            cid, data_iterable = item
            company_info_payload = None
            person_info_payloads = []

            for tag, payload in data_iterable:
                if tag == 'company_match_info':
                    company_info_payload = payload
                elif tag == 'person_match_info':
                    person_info_payloads.append(payload)

            results = []
            if company_info_payload and person_info_payloads:  # Inner join logic
                for p_info in person_info_payloads:
                    results.append((f"match_{p_info[0]}_{cid}", {
                        'person_id': p_info[0], 'person_name': p_info[1], 'person_surname': p_info[2],
                        'company_id': cid, 'company_name': company_info_payload[0],
                        'company_sector': company_info_payload[1],
                        'match_type': 'person_and_company'
                    }))
            return results

        # RDD of (unique_key, {details for matched person-company})
        matched_rdd = grouped_for_match.flatMap(process_rdd12_match_group)

        # Unemployed persons (persons in self.persons not in any employment)
        unemployed_persons_rdd = persons_full_rdd.filter(lambda p_row_tuple: not p_row_tuple[1]['employments']) \
            .map(lambda p_row_tuple: (f"unemployed_{p_row_tuple[0]}", {
            'person_id': p_row_tuple[0], 'person_name': p_row_tuple[1]['name'],
            'person_surname': p_row_tuple[1]['surname'],
            'company_id': None, 'company_name': None, 'company_sector': None,
            'match_type': 'unemployed_person'
        }))

        employments_flat_rdd = persons_full_rdd.flatMap(
            lambda p_row_tuple: [(emp.company_id, p_row_tuple[0]) for emp in p_row_tuple[1]['employments']] if
            p_row_tuple[1]['employments'] else []
        )

        companies_with_employees_marker = employments_flat_rdd.map(lambda x: (x[0], 'has_employees_marker')).distinct()

        all_companies_tagged = companies_full_rdd.map(lambda c: (c[0], ('company_data_for_no_emp_check', c[1])))

        # Union companies data with employee markers, then group
        company_employment_check_grouped = all_companies_tagged.union(
            companies_with_employees_marker.map(lambda x: (x[0], (x[1], None)))
            # Tag needs two elements for consistency
        ).groupByKey()

        def find_companies_without_employees(item):
            cid, data_iterable = item
            company_row = None
            has_employees = False
            for tag, payload in data_iterable:
                if tag == 'company_data_for_no_emp_check':
                    company_row = payload
                elif tag == 'has_employees_marker':
                    has_employees = True

            if company_row and not has_employees:
                return [(f"company_no_emp_{cid}", {
                    'person_id': None, 'person_name': None, 'person_surname': None,
                    'company_id': cid, 'company_name': company_row['name'], 'company_sector': company_row['sector'],
                    'match_type': 'company_without_employees'
                })]
            return []

        companies_no_employees_rdd = company_employment_check_grouped.flatMap(find_companies_without_employees)
        return matched_rdd.union(unemployed_persons_rdd).union(companies_no_employees_rdd)


if __name__ == "__main__":
    # Connection configs
    pg_cfg = {"user": "postgres", "password": "15022001", "database": "mydb",
              "host": "localhost", "port": 5432}
    mongo_cfg = {"host": "localhost", "port": 27017}

    runner = QueryRunner(pg_cfg, mongo_cfg)
    runner.clear_tables()

    # Define test scenarios: (num_persons, num_companies, employment_rate)
    scenarios = [(p, c, r) for p in [100, 1000, 10000]
                             for c in [10, 100]
                             for r in [0.1, 0.25, 0.5]]

    results = {}
    # Execute queries and record timings
    for persons, comps, rate in scenarios:
        label = f"P{persons}_C{comps}_{int(rate*100)}pct"
        print(f"=== Running scenario {label} ===")
        runner.clear_tables()
        runner.generate_data(persons, comps, rate)
        runner.load_spark()

        times = {}
        # SQL timings
        for name, sql in runner.sql_queries.items():
            times[name] = runner.time_sql(sql)
        # DataFrame timings
        for idx, func in enumerate([runner.df5, runner.df6, runner.df7, runner.df8], start=5):
            times[f"--> Part1 Exercise {idx}"] = runner.time_df(func())
        # RDD timings
        for idx, func in enumerate([runner.rdd9, runner.rdd10, runner.rdd11, runner.rdd12], start=9):
            times[f"--> Part1 Exercise {idx}"] = runner.time_rdd(func())

        results[label] = times
        for k, v in times.items():
            print(f" {k:<12}: {v:.6f}s")
        print()

    # Export results to Excel
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = 'Scenario'
    excel_filename = '../Database/performance_results.xlsx'
    try:
        df_results.to_excel(excel_filename)
        print(f"\nResults successfully exported to {excel_filename}")
    except Exception as e:
        print(f"\nError exporting to Excel: {e}")

    def plot_comparison(results_dict, normalize=True):
        df_rows = []
        for scenario_label, query_times in results_dict.items():
            parts = scenario_label.split('_')
            num_persons = int(parts[0][1:])
            num_companies = int(parts[1][1:])
            occupancy_rate = int(parts[2][:-3]) / 100.0

            row = {'Scenario': scenario_label,
                   'Persons': num_persons,
                   'Companies': num_companies,
                   'Rate': occupancy_rate}
            row.update(query_times)
            df_rows.append(row)

        df = pd.DataFrame(df_rows)
        df = df.sort_values(by=['Persons', 'Companies', 'Rate'])

        query_names = [col for col in df.columns if col not in ['Scenario', 'Persons', 'Companies', 'Rate']]

        if normalize:
            for q in query_names:
                min_t, max_t = df[q].min(), df[q].max()
                if max_t > min_t:
                    df[q] = (df[q] - min_t) / (max_t - min_t)
                else:
                    df[q] = 0

        plt.figure(figsize=(15, 8))
        for query in query_names:
            plt.plot(df['Scenario'], df[query], marker='o', label=query)

        plt.xlabel('Scenario (Persons_Companies_EmploymentRate)')
        plt.ylabel('Normalized Time' if normalize else 'Time (s)')
        plt.title('Performance Comparison Across Scenarios' + (' (Normalized)' if normalize else ''))
        plt.xticks(rotation=60, ha='right')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def save_to_excel(results_dict, filename="query_results.xlsx"):
        """Save the performance results to an Excel file."""
        df_rows = []
        for scenario_label, query_times in results_dict.items():
            parts = scenario_label.split('_')
            num_persons = int(parts[0][1:])
            num_companies = int(parts[1][1:])
            occupancy_rate = int(parts[2][:-3]) / 100.0

            row = {
                'Scenario': scenario_label,
                'Persons': num_persons,
                'Companies': num_companies,
                'Employment Rate': occupancy_rate
            }
            row.update(query_times)
            df_rows.append(row)

        df = pd.DataFrame(df_rows)
        cols = ['Scenario', 'Persons', 'Companies', 'Employment Rate'] + \
               [c for c in df.columns if c not in ['Scenario', 'Persons', 'Companies', 'Employment Rate']]
        df = df[cols].sort_values(by=['Persons', 'Companies', 'Employment Rate'])
        df.to_excel(filename, index=False)
        print(f"\nResults saved to {filename}")


    save_to_excel(results, "performance_results_part1.xlsx")
    plot_comparison(results, normalize=True)