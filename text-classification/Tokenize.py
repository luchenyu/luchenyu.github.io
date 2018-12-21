import tokenization

class MyTokenizer(tokenization.FullTokenizer):

    def __init__(self, data):
        self.vocab = {}
        self.inv_vocab = {}
        index = 0
        for line in data.split('\n'):
            token = tokenization.convert_to_unicode(line.strip())
            self.vocab[token] = index
            self.inv_vocab[index] = token
            index += 1
        tokenization.FullTokenizer.__init__(self, do_lower_case=True)

def tokenize(text, tokenizer):
	tokens = tokenizer.tokenize(text)
	tokens.insert(0, '[CLS]')
	tokens.append('[SEP]')
	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	return input_ids
