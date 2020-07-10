
import ai_benchmark


def main():
    benchmark = ai_benchmark.AIBenchmark()
    results = benchmark.run_training(precision="high")



if __name__ == "__main__":
    # execute only if run as a script
    main()
