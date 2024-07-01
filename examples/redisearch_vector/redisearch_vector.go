package main

import (
	"context"
	"fmt"
	"github.com/RediSearch/redisearch-go/v2/redisearch"
	"github.com/gomodule/redigo/redis"
	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/sashabaranov/go-openai"
	"log"
	"strings"
	"time"
)

// Constants for configuration
const (
	INDEX_NAME = "MY_INDEX"
	DIMS       = 1536 // Dimensions for the vector embeddings
	DB         = 0    // Redis database number
	API_KEY    = "SOME_OPENAI_API_KEY"
	TIMEOUT    = 30 * time.Second
	ADDRESS    = "localhost:6379"
	PASSWORD   = "secrete_password"
)

// InitRedis initializes and returns a Redis client
func InitRedis() (*redisearch.Client, error) {
	pool := &redis.Pool{
		Dial: func() (redis.Conn, error) {
			return redis.Dial("tcp", ADDRESS,
				redis.DialPassword(PASSWORD),
				redis.DialDatabase(DB),
			)
		},
	}

	client := redisearch.NewClientFromPool(pool, INDEX_NAME)
	return client, nil
}

// CreateSchema sets up the Redis search index schema
func CreateSchema(database *redisearch.Client) error {
	sc := redisearch.NewSchema(redisearch.DefaultOptions).
		AddField(redisearch.NewTextFieldOptions("chat", redisearch.TextFieldOptions{NoStem: true, NoIndex: true})).
		AddField(redisearch.NewVectorFieldOptions("chat_embeddings", redisearch.VectorFieldOptions{
			Algorithm: redisearch.Flat,
			Attributes: map[string]interface{}{
				"DIM":             DIMS,
				"DISTANCE_METRIC": "COSINE",
				"TYPE":            "FLOAT32",
			},
		}))

	_ = database.Drop()

	if err := database.CreateIndex(sc); err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	return nil
}

// AddVector adds a new vector to the Redis index
func AddVector(chat string, database *redisearch.Client) error {
	key := uuid.New().String()
	vector, err := ApiEmbedding(chat)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	doc := redisearch.NewDocument(key, 1.0)
	doc.Set("chat", chat).
		Set("chat_embeddings", rueidis.VectorString32(vector))

	if err := database.Index([]redisearch.Document{doc}...); err != nil {
		return fmt.Errorf("failed to index document: %w", err)
	}

	return nil
}

// ApiEmbedding generates an embedding for the given input using OpenAI API
func ApiEmbedding(input string) ([]float32, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return nil, fmt.Errorf("input is empty")
	}

	client := openai.NewClient(API_KEY)
	ctx, cancel := context.WithTimeout(context.Background(), TIMEOUT)
	defer cancel()

	req := openai.EmbeddingRequest{
		Input: []string{input},
		Model: openai.AdaEmbeddingV2,
	}

	resp, err := client.CreateEmbeddings(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("error creating embeddings: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embeddings found for the input")
	}

	embedding := resp.Data[0].Embedding
	return embedding, nil
}

// AddData populates the Redis index with sample data
func AddData(client *redisearch.Client) error {
	// ... [sample data]
	data := []struct {
		category string
		texts    []string
	}{
		{
			category: "Technology and Innovation",
			texts: []string{
				"Artificial intelligence is transforming industries by automating tasks and enhancing decision-making.",
				"The rise of virtual reality has opened new possibilities for remote learning and immersive gaming.",
				"Self-driving cars are poised to revolutionize the way we commute, reducing accidents and traffic congestion.",
				"Wearable technology, like smartwatches, is making it easier for people to monitor their health metrics in real time.",
				"Advances in renewable energy technologies are crucial for combating climate change and reducing global reliance on fossil fuels.",
			},
		},
		{
			category: "Travel and Exploration",
			texts: []string{
				"Backpacking across Europe offers an intimate glimpse into the rich history and diverse cultures of the continent.",
				"The popularity of eco-tourism is helping to preserve natural habitats while providing travelers with unique experiences.",
				"Exploring the coral reefs of Australia is a breathtaking adventure that highlights the beauty of marine biodiversity.",
				"Historical landmarks, from the Great Wall of China to the pyramids of Egypt, tell stories of ancient civilizations.",
				"Culinary tours in cities like Paris and Tokyo allow travelers to explore the local flavors and culinary traditions.",
			},
		},
		{
			category: "Health and Wellness",
			texts: []string{
				"Regular exercise is key to maintaining physical health and improving overall well-being.",
				"Mental health is gaining recognition as a critical aspect of overall health, with mindfulness and meditation becoming more popular.",
				"The benefits of a balanced diet are well-documented, including improved energy levels and better immune system function.",
				"Sleep hygiene plays a crucial role in physical and mental health, impacting mood, cognition, and performance.",
				"The rise of telemedicine is making healthcare more accessible, allowing patients to consult with doctors remotely.",
			},
		},
		{
			category: "Education and Learning",
			texts: []string{
				"Online education platforms are expanding access to learning opportunities for people around the world.",
				"The integration of technology in classrooms is enhancing interactive learning and engagement among students.",
				"Lifelong learning is essential for career development and staying current with evolving industry trends.",
				"Critical thinking and problem-solving are fundamental skills that education systems aim to instill in students.",
				"Bilingual education has numerous cognitive benefits, including better memory and enhanced problem-solving skills.",
			},
		},
	}

	for _, category := range data {
		fmt.Printf("Adding data for category: %s\n", category.category)
		for _, text := range category.texts {
			if err := AddVector(text, client); err != nil {
				return fmt.Errorf("failed to add data to vector cache for category %s: %w", category.category, err)
			}
			time.Sleep(time.Second * 2) // Reduced sleep time for faster execution
		}
	}

	fmt.Println("All data loaded successfully!")
	return nil
}

// SearchInVectorCache performs a KNN search in the Redis index
func SearchInVectorCache(userQueryEmbedding []float32, database *redisearch.Client) (docs []redisearch.Document, total int, err error) {
	maxLimit := 5
	userQueryParsed := rueidis.VectorString32(userQueryEmbedding)

	r := redisearch.Query{
		Raw: fmt.Sprintf("(*)=>[KNN 10 @chat_embeddings $query_vector AS vector_dist]"),
		Params: map[string]interface{}{
			"query_vector": userQueryParsed,
		},
		Dialect: 2,
		SortBy: &redisearch.SortingKey{
			Field: "vector_dist",
		},
		ReturnFields: []string{"chat", "vector_dist"},
	}
	query := r.Limit(0, maxLimit)
	docs, total, err = database.Search(query)
	if err != nil {
		return nil, -1, fmt.Errorf("failed to perform search: %w", err)
	}

	return
}

func main() {
	client, err := InitRedis()
	if err != nil {
		log.Fatalf("Failed to initialize Redis client: %v", err)
	}
	fmt.Println("Connection established successfully!")

	if err := CreateSchema(client); err != nil {
		log.Fatalf("Failed to create chat schema: %v", err)
	}
	fmt.Println("Schema creation successful!")

	if err := AddData(client); err != nil {
		log.Fatalf("Failed to add data: %v", err)
	}

	fmt.Println("Data Added Successfully ..!!")

	queryText := "Wearable technology, like smartwatches, is making it easier for people"
	vec, err := ApiEmbedding(queryText)
	if err != nil {
		log.Fatalf("Failed to create embedding: %v", err)
	}

	docs, total, err := SearchInVectorCache(vec, client)
	if err != nil {
		log.Fatalf("Failed to search in vector cache: %v", err)
	}

	fmt.Println("Search Results:")
	fmt.Println("Total Results: ", total)
	for i, doc := range docs {
		fmt.Printf("%d. %s\n", i+1, doc.Properties["chat"])
		fmt.Printf("%s\n\n", doc.Properties["vector_dist"])
	}
}
