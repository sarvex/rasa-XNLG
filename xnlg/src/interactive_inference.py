import fastBPE
from src.data.dictionary import Dictionary
from src.utils import to_cuda, AttrDict, bool_flag
from src.model.transformer import TransformerModel
from src.evaluation.xsumm import XSumm, XSumm_LANGS, tokens2words
import numpy as np
import torch
import os

BOS_WORD = '<s>'
EOS_WORD = '</s>'
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'

SPECIAL_WORD = '<special%i>'
SPECIAL_WORDS = 10

SEP_WORD = SPECIAL_WORD % 0
MASK_WORD = SPECIAL_WORD % 1


class XNLGParaphraser:

    def __init__(self, codes_path, vocab_path, model_path):

        self.params = {'decode_vocab_sizes': "95000, 95000",
                       'beam_size': 3,
                       'max_dec_len': 32
                       }
        self.bpe = self.load_bpe(codes_path, vocab_path)
        # self.dico = Dictionary.read_vocab(vocab_path)
        self.encoder, self.decoder, self.dico = self.load_model(model_path, self.params)

    @staticmethod
    def load_bpe(codes_path, vocab_path):
        bpe = fastBPE.fastBPE(codes_path, vocab_path)
        return bpe

    def apply_bpe(self, sentences):
        return self.bpe.apply(sentences)

    def load_model(self, model_path, params):
        # check parameters
        reloaded = torch.load(model_path)

        encoder_model_params = AttrDict(reloaded['enc_params'])
        decoder_model_params = AttrDict(reloaded['dec_params'])

        dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])

        params['n_langs'] = encoder_model_params['n_langs']
        params['id2lang'] = encoder_model_params['id2lang']
        params['lang2id'] = encoder_model_params['lang2id']
        params['n_words'] = len(dico)
        params['bos_index'] = dico.index(BOS_WORD)
        params['eos_index'] = dico.index(EOS_WORD)
        params['pad_index'] = dico.index(PAD_WORD)
        params['unk_index'] = dico.index(UNK_WORD)
        params['mask_index'] = dico.index(MASK_WORD)

        encoder = TransformerModel(encoder_model_params, dico, is_encoder=True, with_output=False)
        decoder = TransformerModel(decoder_model_params, dico, is_encoder=False, with_output=True)

        def _process_state_dict(state_dict):
            return {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

        encoder.load_state_dict(_process_state_dict(reloaded['encoder']))
        decoder.load_state_dict(_process_state_dict(reloaded['decoder']))

        return encoder, decoder, dico

    def index_data(self, original_sentences):
        """
        Index sentences with a dictionary.
        """

        positions = []
        sentences = []
        unk_words = {}

        for s in original_sentences:
            count_unk = 0
            indexed = []
            for w in s:
                word_id = self.dico.index(w, no_unk=False)
                # if we find a special word which is not an unknown word, skip the sentence
                if 0 <= word_id < 4 + SPECIAL_WORDS and word_id != 3:
                    # logger.warning('Found unexpected special word "%s" (%i)!!' % (w, word_id))
                    continue
                assert word_id >= 0
                indexed.append(word_id)
                if word_id == self.dico.unk_index:
                    unk_words[w] = unk_words.get(w, 0) + 1
                    count_unk += 1
            # add sentence
            positions.append([len(sentences), len(sentences) + len(indexed)])
            sentences.extend(indexed)
            sentences.append(1)  # EOS index

        # tensorize data
        positions = np.int64(positions)
        if len(self.dico) < 1 << 16:
            sentences = np.uint16(sentences)
        elif len(self.dico) < 1 << 31:
            sentences = np.int32(sentences)
        else:
            raise Exception("Dictionary is too big.")
        assert sentences.min() >= 0
        data = {
            'dico': self.dico,
            'positions': positions,
            'sentences': sentences,
            'unk_words': unk_words,
        }

        return data

    def batch_sentences(self, sentences):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.params['pad_index'])

        sent[0] = self.params['eos_index']
        for i, s in enumerate(sentences):
            if lengths[i] > 2:  # if sentence not empty
                sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            sent[lengths[i] - 1, i] = self.params['eos_index']
        return sent, lengths


    def preprocess(self, sentences):

        tokenized_sentences = self.apply_bpe(sentences)
        preprocessed_data = self.index_data(tokenized_sentences)
        return preprocessed_data

    def model_inference(self, direction, batch):
        direction = direction.split("-")
        params = self.params
        encoder = self.encoder
        decoder = self.decoder
        # encoder.eval()
        # decoder.eval()
        dico = self.dico

        x_lang, y_lang = direction
        # print("Performing %s-%s-xsumm" % (x_lang, y_lang))

        X, Y = [], []
        x_lang_id = params['lang2id'][x_lang[-2:]]
        y_lang_id = params['lang2id'][y_lang[-2:]]
        vocab_mask = None

        sent_x, len_x = batch
        lang_x = sent_x.clone().fill_(x_lang_id)

        sent_x, len_x, lang_x = to_cuda(sent_x, len_x, lang_x)

        with torch.no_grad():
            encoded = encoder(
                "fwd", x=sent_x, lengths=len_x, langs=lang_x, causal=False)
            encoded = encoded.transpose(0, 1)

            if params['beam_size'] == 1:
                decoded, _ = decoder.generate(
                    encoded, len_x, y_lang_id, max_len=params['max_dec_len'],
                    vocab_mask=vocab_mask)
            else:
                decoded, _ = decoder.generate_beam(
                    encoded, len_x, y_lang_id, beam_size=params['beam_size'],
                    length_penalty=0.9, early_stopping=False,
                    max_len=params['max_dec_len'], vocab_mask=vocab_mask)

        for j in range(decoded.size(1)):
            sent = decoded[:, j]
            delimiters = (sent == params['eos_index']).nonzero().view(-1)
            assert len(delimiters) >= 1 and delimiters[0].item() == 0
            sent = sent[1:] if len(delimiters) == 1 else sent[1: delimiters[1]]

            trg_tokens = [dico[sent[k].item()] for k in range(len(sent))]
            trg_words = tokens2words(trg_tokens)
            if y_lang.endswith("zh"):
                Y.append(" ".join("".join(trg_words)))
            else:
                Y.append(" ".join(trg_words))

        return Y

    def gen_paraphrases(self, sentence, direction="en-en"):

        preprocessed_sentence = self.preprocess([sentence])
        batch = self.batch_sentences(preprocessed_sentence['sentences'])
        y = self.model_inference(direction, batch)

        return y


if __name__ == '__main__':

    codes_path = ''
    vocab_path = ''
    model_path = ''
    sentence = 'What are some good restaurants serving pizza here?'
    direction = 'en-en'
    pp = XNLGParaphraser(codes_path, vocab_path, model_path)
    pp.gen_paraphrases(sentence, direction)



