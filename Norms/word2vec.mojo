from Random import rand
from Math import exp, log
from Time import now
from Vector import DynamicVector, InlinedFixedVector
from Dict import Dict
from String import join

struct UnigamTable:
    var table: DTypePointer[DType.int32]
    var size: Int
    
    fn __init__(inout self, vocab_size: Int, freq: DTypePointer[DType.float64], table_size: Int = 100000000):
        self.size = table_size
        self.table = DTypePointer[DType.int32].alloc(table_size)
        
        var total_words: Float64 = 0.0
        for i in range(vocab_size):
            total_words += freq[i]
        
        var cumulative: Float64 = 0.0
        var i: Int = 0
        
        for word_idx in range(vocab_size):
            cumulative += freq[word_idx]
            var target = (cumulative / total_words) * table_size
            
            while Float64(i) < target and i < table_size:
                self.table[i] = word_idx
                i += 1
    
    fn __del__(owned self):
        self.table.free()
    
    fn sample(self) -> Int:
        return self.table[Int(rand(Float64(self.size)))]

struct Word2Vec:
    # Model hyperparameters
    var vector_size: Int
    var window_size: Int
    var min_count: Int
    var negative_samples: Int
    var learning_rate: Float64
    var epochs: Int
    
    # Model state
    var vocab_size: Int
    var word_vectors: DTypePointer[DType.float64]  # word_vectors[word_idx * vector_size + i]
    var context_vectors: DTypePointer[DType.float64]  # context_vectors[word_idx * vector_size + i]
    var word_freqs: DTypePointer[DType.float64]
    var unigram_table: UnigamTable
    var vocab_map: Dict[String, Int]
    var vocab_list: DynamicVector[String]
    
    fn __init__(inout self, vector_size: Int = 100, window_size: Int = 5, 
                min_count: Int = 5, negative_samples: Int = 5, 
                learning_rate: Float64 = 0.025, epochs: Int = 5):
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.vocab_size = 0
        self.word_vectors = DTypePointer[DType.float64].alloc(0)
        self.context_vectors = DTypePointer[DType.float64].alloc(0)
        self.word_freqs = DTypePointer[DType.float64].alloc(0)
        self.unigram_table = UnigamTable(0, self.word_freqs)
        self.vocab_map = Dict[String, Int]()
        self.vocab_list = DynamicVector[String]()
    
    fn __del__(owned self):
        self.word_vectors.free()
        self.context_vectors.free()
        self.word_freqs.free()
    
    fn build_vocabulary(inout self, corpus: DynamicVector[String]) -> Int:
        # Count word frequencies
        var word_counts = Dict[String, Int]()
        
        for sentence in corpus:
            var words = sentence.split()
            for word in words:
                if word_counts.contains(word):
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        
        # Filter by minimum count
        var filtered_vocab = DynamicVector[(String, Int)]()
        for word in word_counts.keys():
            let count = word_counts[word]
            if count >= self.min_count:
                filtered_vocab.push_back((word, count))
        
        # Sort by frequency (decreasing)
        # Simple bubble sort
        for i in range(len(filtered_vocab)):
            for j in range(0, len(filtered_vocab) - i - 1):
                if filtered_vocab[j].1 < filtered_vocab[j+1].1:
                    var temp = filtered_vocab[j]
                    filtered_vocab[j] = filtered_vocab[j+1]
                    filtered_vocab[j+1] = temp
        
        # Build vocabulary
        self.vocab_size = len(filtered_vocab)
        self.vocab_map = Dict[String, Int]()
        self.vocab_list = DynamicVector[String]()
        
        # Free old arrays and allocate new ones
        self.word_vectors.free()
        self.context_vectors.free()
        self.word_freqs.free()
        
        self.word_vectors = DTypePointer[DType.float64].alloc(self.vocab_size * self.vector_size)
        self.context_vectors = DTypePointer[DType.float64].alloc(self.vocab_size * self.vector_size)
        self.word_freqs = DTypePointer[DType.float64].alloc(self.vocab_size)
        
        # Initialize vectors with small random values
        for i in range(self.vocab_size * self.vector_size):
            self.word_vectors[i] = (rand(2.0) - 1.0) * 0.01  # Initialize between -0.01 and 0.01
            self.context_vectors[i] = (rand(2.0) - 1.0) * 0.01
        
        for i in range(self.vocab_size):
            let word = filtered_vocab[i].0
            let count = filtered_vocab[i].1
            
            self.vocab_map[word] = i
            self.vocab_list.push_back(word)
            self.word_freqs[i] = Float64(count)
        
        # Build unigram table for negative sampling
        self.unigram_table = UnigamTable(self.vocab_size, self.word_freqs)
        
        return self.vocab_size
    
    fn sigmoid(x: Float64) -> Float64:
        if x > 10.0:  # Avoid overflow
            return 1.0
        elif x < -10.0:
            return 0.0
        return 1.0 / (1.0 + exp(-x))
    
    fn train_pair(inout self, word_idx: Int, context_idx: Int, label: Float64):
        # Get word vector
        var word_vec = InlinedFixedVector[100, Float64]()
        for i in range(self.vector_size):
            word_vec[i] = self.word_vectors[word_idx * self.vector_size + i]
        
        # Get context vector
        var context_vec = InlinedFixedVector[100, Float64]()
        for i in range(self.vector_size):
            context_vec[i] = self.context_vectors[context_idx * self.vector_size + i]
        
        # Calculate prediction (dot product)
        var dot: Float64 = 0.0
        for i in range(self.vector_size):
            dot += word_vec[i] * context_vec[i]
        
        # Calculate error
        let pred = sigmoid(dot)
        let error = label - pred
        
        # Update vectors
        for i in range(self.vector_size):
            # Update word vector
            let word_grad = error * context_vec[i] * self.learning_rate
            self.word_vectors[word_idx * self.vector_size + i] += word_grad
            
            # Update context vector
            let context_grad = error * word_vec[i] * self.learning_rate
            self.context_vectors[context_idx * self.vector_size + i] += context_grad
    
    fn train(inout self, corpus: DynamicVector[String]) -> Float64:
        if self.vocab_size == 0:
            print("Error: Build vocabulary first")
            return -1.0
        
        var total_words: Int = 0
        var start_time = now()
        
        for epoch in range(self.epochs):
            print("Starting epoch", epoch + 1, "/", self.epochs)
            
            for sentence_idx in range(len(corpus)):
                let sentence = corpus[sentence_idx]
                var words = sentence.split()
                
                for pos in range(len(words)):
                    let word = words[pos]
                    
                    if not self.vocab_map.contains(word):
                        continue  # Skip OOV words
                        
                    let word_idx = self.vocab_map[word]
                    
                    # Dynamic window size: randomly sample from [1, window_size]
                    let current_window = Int(rand(Float64(self.window_size))) + 1
                    
                    # Train on window contexts
                    for offset in range(-current_window, current_window + 1):
                        if offset == 0:
                            continue  # Skip the word itself
                            
                        let context_pos = pos + offset
                        if context_pos < 0 or context_pos >= len(words):
                            continue  # Skip out-of-bounds
                            
                        let context_word = words[context_pos]
                        if not self.vocab_map.contains(context_word):
                            continue
                            
                        let context_idx = self.vocab_map[context_word]
                        
                        # Train on the positive sample
                        self.train_pair(word_idx, context_idx, 1.0)
                        
                        # Train on negative samples
                        for n in range(self.negative_samples):
                            var neg_idx = self.unigram_table.sample()
                            
                            # Ensure negative sample is not the target word
                            while neg_idx == context_idx:
                                neg_idx = self.unigram_table.sample()
                                
                            # Train on the negative sample
                            self.train_pair(word_idx, neg_idx, 0.0)
                            
                    total_words += 1
                    
                    # Progress reporting
                    if total_words % 10000 == 0:
                        let elapsed_ms = (now() - start_time) / 1_000_000
                        let words_per_sec = Float64(total_words) / (Float64(elapsed_ms) / 1000.0)
                        print("Processed", total_words, "words,", words_per_sec, "words/sec")
        
        let total_time = (now() - start_time) / 1_000_000  # ms
        let words_per_sec = Float64(total_words) / (Float64(total_time) / 1000.0)
        
        print("Training completed in", total_time / 1000.0, "seconds")
        print("Average:", words_per_sec, "words/sec")
        
        return words_per_sec
    
    fn get_vector(self, word: String) -> InlinedFixedVector[100, Float64]:
        var result = InlinedFixedVector[100, Float64]()
        
        if not self.vocab_map.contains(word):
            print("Warning: Word", word, "not in vocabulary")
            return result
            
        let word_idx = self.vocab_map[word]
        
        for i in range(self.vector_size):
            result[i] = self.word_vectors[word_idx * self.vector_size + i]
            
        return result
    
    fn cosine_similarity(x: InlinedFixedVector[100, Float64], y: InlinedFixedVector[100, Float64]) -> Float64:
        var dot: Float64 = 0.0
        var norm_x: Float64 = 0.0
        var norm_y: Float64 = 0.0
        
        for i in range(len(x)):
            dot += x[i] * y[i]
            norm_x += x[i] * x[i]
            norm_y += y[i] * y[i]
            
        if norm_x == 0.0 or norm_y == 0.0:
            return 0.0
            
        return dot / (norm_x.sqrt() * norm_y.sqrt())
    
    fn most_similar(self, word: String, top_n: Int = 5) -> DynamicVector[(String, Float64)]:
        var result = DynamicVector[(String, Float64)]()
        
        if not self.vocab_map.contains(word):
            print("Error: Word", word, "not in vocabulary")
            return result
            
        let word_vec = self.get_vector(word)
        
        # Calculate similarities with all other words
        var similarities = DynamicVector[(String, Float64)]()
        
        for i in range(self.vocab_size):
            let other_word = self.vocab_list[i]
            
            if other_word == word:
                continue
                
            var other_vec = InlinedFixedVector[100, Float64]()
            for j in range(self.vector_size):
                other_vec[j] = self.word_vectors[i * self.vector_size + j]
                
            let similarity = cosine_similarity(word_vec, other_vec)
            similarities.push_back((other_word, similarity))
        
        # Sort by similarity (bubble sort)
        for i in range(len(similarities)):
            for j in range(0, len(similarities) - i - 1):
                if similarities[j].1 < similarities[j+1].1:
                    var temp = similarities[j]
                    similarities[j] = similarities[j+1]
                    similarities[j+1] = temp
        
        # Return top N
        let n = min(top_n, len(similarities))
        for i in range(n):
            result.push_back(similarities[i])
            
        return result

# Example usage with a tiny corpus
fn main():
    print("Word2Vec Example in Mojo")
    
    # Create a small corpus
    var corpus = DynamicVector[String]()
    corpus.push_back("the quick brown fox jumps over the lazy dog")
    corpus.push_back("the dog barked at the fox and the fox ran away")
    corpus.push_back("the lazy dog slept in the sun")
    corpus.push_back("the quick brown fox jumps over the fence")
    corpus.push_back("quick foxes jump over lazy dogs")
    
    # Create and train the model
    var w2v = Word2Vec(vector_size=10, window_size=2, min_count=1, 
                      negative_samples=2, learning_rate=0.025, epochs=5)
    
    print("Building vocabulary...")
    let vocab_size = w2v.build_vocabulary(corpus)
    print("Vocabulary size:", vocab_size)
    
    print("Vocabulary:")
    for i in range(w2v.vocab_size):
        print("  ", w2v.vocab_list[i], ":", w2v.word_freqs[i])
    
    print("\nTraining model...")
    let words_per_sec = w2v.train(corpus)
    
    # Test some similarities
    let test_words = ["fox", "dog", "quick"]
    
    for word in test_words:
        print("\nWords similar to '" + word + "':")
        let similar = w2v.most_similar(word, 3)
        for i in range(len(similar)):
            print("  ", similar[i].0, ":", similar[i].1)
    
    print("\nNote: With this tiny corpus, results may not be meaningful.")
    print("In practice, Word2Vec requires large corpora (millions of words)")
    print("for good semantic representations.")