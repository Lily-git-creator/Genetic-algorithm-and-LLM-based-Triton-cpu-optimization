import argparse
from evoluter import ParaEvoluter, GAEvoluter, DEEvoluter

def main():
    parser = argparse.ArgumentParser(description="LLM Code Evolution Framework")
    
    # 策略选择
    parser.add_argument("--mode", type=str, choices=["para", "ga", "de"], default="para", 
                        help="Evolution Strategy: 'para' (Paraphrasing), 'ga' (Genetic Algo), 'de' (Differential Evo)")
    
    # 超参数
    parser.add_argument("--budget", type=int, default=5, help="Number of generations")
    parser.add_argument("--pop_size", type=int, default=4, help="Population size")
    parser.add_argument("--time_limit", type=int, default=1200, help="Time limit in miliseconds for the evolution process")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Evolution | Mode: {args.mode.upper()} | Pop Size: {args.pop_size}")
    
    # 策略分发
    if args.mode == "para":
        evolver = ParaEvoluter(args)
    elif args.mode == "ga":
        evolver = GAEvoluter(args)
    elif args.mode == "de":
        evolver = DEEvoluter(args)
    
    evolver.run()

if __name__ == "__main__":
    main()
