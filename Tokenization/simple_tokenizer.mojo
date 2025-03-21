from String import join, concat
from Vector import DynamicVector
from Time import now
from Dict import Dict
from IO import File, read_file, write_file
from Path import Path

struct SimpleTokenizer:
    var vocabulary: Dict[String, Int]
    var inverse_vocab: DynamicVector[String]
    var max_tokens: Int
    
    fn __init__(inout self, max_tokens: Int = 50000):
        self.vocabulary = Dict[String, Int]()
        self.inverse_vocab = DynamicVector[String]()
        self.max_tokens = max_tokens
        
        # Add special tokens
        self._add_token("<PAD>")  # Padding token
        self._add_token("<UNK>")  # Unknown token
        self._add_token("<BOS>")  # Beginning of sequence
        self._add_token("<EOS>")  # End of sequence
    
    fn _add_token(inout self, token: String) -> Int:
        if self.vocabulary.contains(token):
            return self.vocabulary[token]
        
        let token_id = len(self.vocabulary)
        
        if token_id < self.max_tokens:
            self.vocabulary[token] = token_id
            self.inverse_vocab.push_back(token)
            return token_id
        
        return self.vocabulary["<UNK>"]
    
    fn build_vocabulary(inout self, text: String) -> Int:
        let words = text.split()
        var unique_words = Dict[String, Int]()
        
        # Count word frequencies
        for word in words:
            if unique_words.contains(word):
                unique_words[word] += 1
            else:
                unique_words[word] = 1
        
        # Sort by frequency (decreasing)
        var word_counts = DynamicVector[(String, Int)]()
        for key in unique_words.keys():
            word_counts.push_back((key, unique_words[key]))
        
        # Simple bubble sort by count (decreasing)
        for i in range(len(word_counts)):
            for j in range(len(word_counts) - i - 1):
                if word_counts[j].1 < word_counts[j+1].1:
                    let temp = word_counts[j]
                    word_counts[j] = word_counts[j+1]
                    word_counts[j+1] = temp
        
        # Add tokens to vocabulary
        var added_count = 0
        for i in range(len(word_counts)):
            if len(self.vocabulary) >= self.max_tokens:
                break
                
            let word = word_counts[i].0
            if not self.vocabulary.contains(word):
                self._add_token(word)
                added_count += 1
        
        return added_count
    
    fn tokenize(self, text: String) -> DynamicVector[Int]:
        var tokens = DynamicVector[Int]()
        let words = text.split()
        
        tokens.push_back(self.vocabulary["<BOS>"])
        
        for word in words:
            if self.vocabulary.contains(word):
                tokens.push_back(self.vocabulary[word])
            else:
                tokens.push_back(self.vocabulary["<UNK>"])
        
        tokens.push_back(self.vocabulary["<EOS>"])
        
        return tokens
    
    fn detokenize(self, token_ids: DynamicVector[Int]) -> String:
        var words = DynamicVector[String]()
        
        for i in range(len(token_ids)):
            let token_id = token_ids[i]
            
            # Skip special tokens
            if token_id == self.vocabulary["<PAD>"] or 
               token_id == self.vocabulary["<BOS>"] or 
               token_id == self.vocabulary["<EOS>"]:
                continue
                
            if token_id >= 0 and token_id < len(self.inverse_vocab):
                words.push_back(self.inverse_vocab[token_id])
            else:
                words.push_back("<UNK>")
        
        return " ".join(words)
    
    fn vocab_size(self) -> Int:
        return len(self.vocabulary)
    
    fn save_vocabulary(self, filepath: String) raises -> Bool:
        var content = ""
        
        for i in range(len(self.inverse_vocab)):
            let token = self.inverse_vocab[i]
            content += token + "\n"
        
        try:
            _ = write_file(filepath, content)
            return True
        except:
            print("Error saving vocabulary to", filepath)
            return False
    
    fn load_vocabulary(inout self, filepath: String) raises -> Bool:
        try:
            let content = read_file(filepath)
            let lines = content.split('\n')
            
            # Clear current vocabulary
            self.vocabulary = Dict[String, Int]()
            self.inverse_vocab = DynamicVector[String]()
            
            for i in range(len(lines)):
                let line = lines[i].strip()
                if len(line) > 0:
                    self._add_token(line)
            
            return True
        except:
            print("Error loading vocabulary from", filepath)
            return False

fn main() raises:
    print("Simple Tokenizer Example in Mojo")
    
    # Sample text
    var text = """
    Mojo is a programming language that combines Python's ease of use with systems programming 
    capabilities for AI and high-performance computing. It is designed to be fast and efficient, 
    making it ideal for machine learning applications. Mojo bridges the gap between research and 
    production by offering both flexibility and performance.
    """
    
    var tokenizer = SimpleTokenizer(1000)
    
    print("Building vocabulary...")
    let added = tokenizer.build_vocabulary(text)
    print("Added", added, "tokens to vocabulary")
    print("Total vocabulary size:", tokenizer.vocab_size())
    
    print("\nTokenizing sample text...")
    let start = now()
    var tokens = tokenizer.tokenize(text)
    let end = now()
    
    print("Tokenization took", (end - start) / 1_000_000, "ms")
    
    print("\nTokens:")
    for i in range(len(tokens)):
        if i > 0:
            print_no_newline(", ")
        print_no_newline(tokens[i])
    print()
    
    print("\nDetokenizing...")
    let reconstructed = tokenizer.detokenize(tokens)
    print("Reconstructed text:\n", reconstructed)
    
    print("\nSaving vocabulary...")
    try:
        _ = tokenizer.save_vocabulary("vocab.txt")
        print("Vocabulary saved successfully")
    except:
        print("Failed to save vocabulary")
    
    print("\nCreating new tokenizer and loading vocabulary...")
    var new_tokenizer = SimpleTokenizer()
    try:
        let success = new_tokenizer.load_vocabulary("vocab.txt")
        if success:
            print("Vocabulary loaded successfully")
            print("Loaded vocabulary size:", new_tokenizer.vocab_size())
        else:
            print("Failed to load vocabulary")
    except:
        print("Exception while loading vocabulary")