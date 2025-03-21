from Worker import Worker
from ReviewDataset import ReviewDataset
from LLMService import LLMService

def main():
    dataset = ReviewDataset("docs\\tripadvisor_hotel_reviews_short.csv")  # Load dataset
    scorer = LLMService(model_path="bert-base-uncased", device="cpu")  # Initialize BERT scorer
    worker = Worker(dataset, scorer)  # Create experiment
    results_df = worker.run()  # Run experiment
    results_df.to_csv("experiment_results.csv", index=False)  # Save results

if __name__ == '__main__':
    main()