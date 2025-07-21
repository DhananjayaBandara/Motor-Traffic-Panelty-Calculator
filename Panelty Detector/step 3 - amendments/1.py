from RAG import MotorTrafficActRAG

def test_rag_system():
    """Test the RAG system with sample queries"""
    
    # Initialize the RAG system
    csv_path = "motor_traffic_act.csv"
    rag = MotorTrafficActRAG(csv_path)
    
    # Test queries
    test_queries = [
        "registration of motor vehicle",
        "driving licence requirements",
        "speed limits for vehicles",
        "insurance requirements",
        "penalties for violations"
    ]
    
    print("=== TESTING MOTOR TRAFFIC ACT RAG SYSTEM ===\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        print("=" * 50)
        
        results = rag.search(query, top_k=2)
        formatted_results = rag.format_results(results)
        print(formatted_results)
        
        print("\n" + "="*80 + "\n")

def interactive_search():
    """Interactive search function"""
    csv_path = "motor_traffic_act.csv"
    rag = MotorTrafficActRAG(csv_path)
    
    print("Interactive Motor Traffic Act Search")
    print("Enter your queries to search the Motor Traffic Act")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("Your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if query:
            results = rag.search(query, top_k=3)
            print("\n" + rag.format_results(results))
        else:
            print("Please enter a valid query.\n")

if __name__ == "__main__":
    # Uncomment the function you want to run
    # test_rag_system()
    interactive_search()
