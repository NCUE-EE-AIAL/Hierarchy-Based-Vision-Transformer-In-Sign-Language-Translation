from transformers import T5Tokenizer

class Tokenizer:

    def __init__(self):
        # Initialize the T5 tokenizer
        self.t5 = T5Tokenizer.from_pretrained('t5-small', legacy=False)

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        text = text.lower()
        token = self.t5.tokenize(text)

        return ["<sos>"] + token + ["<eos>"]



