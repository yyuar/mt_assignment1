from collections import defaultdict
from types import GeneratorType
import dynet as dy
import numpy as np
import random
import sys
from heapq import *

startSymbol = '<S>'
endSymbol = '</S>'
unkSymbol = '<unk>'

class Attention:

	def __init__(self, model, training_src, training_tgt):
		self.model = model
		self.training = [(x, y) for (x, y) in zip(training_src, training_tgt)]
		#get id-token match
		threshold = 3
		#src
		dicS = defaultdict(int)
		for sen in training_src:
			for item in sen:
				dicS[item]+=1
		for k,v in dicS.items():
			if v<threshold:
				del dicS[k]
		self.src_id_to_token = dicS.keys()+[startSymbol, endSymbol, unkSymbol]
		self.src_token_to_id = defaultdict(lambda:self.src_token_to_id[unkSymbol], {c:i for i,c in enumerate(self.src_id_to_token)})
		
		#tgt
		dicT = defaultdict(int)
		for sen in training_tgt:
			for item in sen:
				dicT[item]+=1
		for k,v in dicT.items():
			if v<threshold:
				del dicT[k]
		self.tgt_id_to_token = dicT.keys()+[startSymbol, endSymbol, unkSymbol]
		self.tgt_token_to_id = defaultdict(lambda:self.tgt_token_to_id[unkSymbol], {c:i for i,c in enumerate(self.tgt_id_to_token)})
		
		self.src_vocab_size = len(self.src_id_to_token)
		self.tgt_vocab_size = len(self.tgt_id_to_token)
		self.embed_size = 512
		self.layers = 1
		self.hidden_size = 512
		self.attention_size = 128
		self.max_len = 70 #the max length of a target sentence

		self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
		self.tgt_lookup = model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))
		self.l2r_builder = dy.LSTMBuilder(self.layers, self.embed_size, self.hidden_size, model)
		self.r2l_builder = dy.LSTMBuilder(self.layers, self.embed_size, self.hidden_size, model)

		self.dec_builder = dy.LSTMBuilder(self.layers, self.embed_size + 2*self.hidden_size, self.hidden_size, model)

		self.W_y = model.add_parameters((self.tgt_vocab_size, self.hidden_size))
		self.b_y = model.add_parameters((self.tgt_vocab_size))

		self.W1_att_f = model.add_parameters((self.attention_size, 2*self.hidden_size))
		self.W1_att_e = model.add_parameters((self.attention_size, self.hidden_size))
		self.w2_att = model.add_parameters((self.attention_size))

	# Calculates the context vector using a MLP
	# h_fs: matrix of embeddings for the source words
	# h_e: hidden state of the decoder
	def __attention_mlp(self, h_fs_matrix, h_e):
		W1_att_f = dy.parameter(self.W1_att_f)
		W1_att_e = dy.parameter(self.W1_att_e)
		w2_att = dy.parameter(self.w2_att)
		# Calculate the alignment score vector
		# Hint: Can we make this more efficient?
		a_t = dy.transpose(dy.tanh(dy.colwise_add(W1_att_f * h_fs_matrix, W1_att_e * h_e))) * w2_att
		alignment = dy.softmax(a_t)
		c_t = h_fs_matrix * alignment
		#find the word with highest attention
		probs = alignment.vec_value()
		highAtteID = np.argmax(probs)
		return (c_t,highAtteID)

	# Training step over a single sentence pair
	def step(self, instances):
		dy.renew_cg()

		W_y = dy.parameter(self.W_y)
		b_y = dy.parameter(self.b_y)
		W1_att_f = dy.parameter(self.W1_att_f)
		W1_att_e = dy.parameter(self.W1_att_e)
		w2_att = dy.parameter(self.w2_att)
		
		#instances : a list [(src0,tgt0),(src1,tgt1),(src2,tgt2)]
		maxLen = max(map(lambda x:len(x[1]),instances))
		src_sents = []
		src_sents_rev = []
		tgt_sents = []
		srcSenLen = len(instances[0][0]) + 2  #the length of the src sentence, all the same
		tgtSenLen = maxLen + 1
		masks = [[] for i in range(tgtSenLen)] #mask for each position. each item in this list is a list with length=batchsize
		num_words = 0
		
		for item in instances:
			#item[0]:src ; item[1]:tgt
			num_words += (len(item[1])+1)
			padNum = maxLen - len(item[1])
			for i in range(len(item[1])+1):
				masks[i].append(1)
			for i in range(len(item[1])+1, tgtSenLen):
				masks[i].append(0)
			thisSrc = [startSymbol] + item[0] + [endSymbol]
			src_sents.append(thisSrc)
			src_sents_rev.append(list(reversed(thisSrc)))
			thisTgt = item[1] + [endSymbol for i in range(padNum+1)]
			tgt_sents.append(thisTgt)

		# Bidirectional representations
		l2r_state = self.l2r_builder.initial_state()
		r2l_state = self.r2l_builder.initial_state()
		l2r_contexts = []
		r2l_contexts = []
		for i in range(srcSenLen):
			batchSrc = dy.lookup_batch(self.src_lookup, [self.src_token_to_id[x[i]] for x in src_sents])
			batchSrc_rev = dy.lookup_batch(self.src_lookup, [self.src_token_to_id[x[i]] for x in src_sents_rev])
			l2r_state = l2r_state.add_input(batchSrc)
			r2l_state = r2l_state.add_input(batchSrc_rev)
			l2r_contexts.append(l2r_state.output())
			r2l_contexts.append(r2l_state.output())

		r2l_contexts.reverse() 

		# Combine the left and right representations for every word
		h_fs = []
		for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
			h_fs.append(dy.concatenate([l2r_i, r2l_i]))
		h_fs_matrix = dy.concatenate_cols(h_fs)

		losses = []

		# Decoder
		c_t = dy.vecInput(self.hidden_size * 2)
		start = dy.concatenate([dy.lookup_batch(self.tgt_lookup, [self.tgt_token_to_id['<S>'] for i in tgt_sents]), c_t])
		dec_state = self.dec_builder.initial_state().add_input(start)
		loss = dy.pickneglogsoftmax_batch(W_y * dec_state.output() + b_y,[self.tgt_token_to_id[tgt_sent[0]] for tgt_sent in tgt_sents])
		losses.append(loss)

		for i in range(tgtSenLen-1):
			#cw : item[i] nw:item[i+1]
			h_e = dec_state.output()
			c_t = self.__attention_mlp(h_fs_matrix, h_e)[0]
			# Get the embedding for the current target word
			embed_t = dy.lookup_batch(self.tgt_lookup, [self.tgt_token_to_id[tgt_sent[i]] for tgt_sent in tgt_sents])
			# Create input vector to the decoder
			x_t = dy.concatenate([embed_t, c_t])
			dec_state = dec_state.add_input(x_t)
			loss = dy.pickneglogsoftmax_batch(W_y * dec_state.output() + b_y,[self.tgt_token_to_id[tgt_sent[i+1]] for tgt_sent in tgt_sents])
			thisMask = dy.inputVector(masks[i+1])
			thisMask = dy.reshape(thisMask,(1,),len(instances))
			losses.append(loss * thisMask)
 
		return dy.sum_batches(dy.esum(losses)), num_words

	def translate_sentence(self, sent):
		dy.renew_cg()

		W_y = dy.parameter(self.W_y)
		b_y = dy.parameter(self.b_y)
		W1_att_f = dy.parameter(self.W1_att_f)
		W1_att_e = dy.parameter(self.W1_att_e)
		w2_att = dy.parameter(self.w2_att)

		sent = [startSymbol] + sent + [endSymbol]
		sent_rev = list(reversed(sent))

		# Bidirectional representations
		l2r_state = self.l2r_builder.initial_state()
		r2l_state = self.r2l_builder.initial_state()
		l2r_contexts = []
		r2l_contexts = []
			
			
		for (cw_l2r, cw_r2l) in zip(sent, sent_rev):
			l2r_state = l2r_state.add_input(dy.lookup(self.src_lookup,self.src_token_to_id[cw_l2r]))
			r2l_state = r2l_state.add_input(dy.lookup(self.src_lookup,self.src_token_to_id[cw_r2l]))
			l2r_contexts.append(l2r_state.output())
			r2l_contexts.append(r2l_state.output())
		r2l_contexts.reverse()

		h_fs = []
		for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
			h_fs.append(dy.concatenate([l2r_i, r2l_i]))
		
		h_fs_matrix = dy.concatenate_cols(h_fs)
		
		# Decoder
		trans_sentence1 = [startSymbol]
		trans_sentence2 = [startSymbol]
		cw1 = trans_sentence1[-1]
		cw2 = trans_sentence2[-1]
		#initial context
		c_t = dy.vecInput(self.hidden_size * 2)
		start = dy.concatenate([dy.lookup(self.tgt_lookup, self.tgt_token_to_id[endSymbol]), c_t])
		init_state = self.dec_builder.initial_state().add_input(start)

		def generate_top_n(logProb, state, words, wordID, n):
			if words[-1] == endSymbol:
				yield logProb, words
			h_e = state.output()
			c_t, unkIndex = self.__attention_mlp(h_fs_matrix, h_e)
			embed_t = dy.lookup(self.tgt_lookup, wordID)
			x_t = dy.concatenate([embed_t, c_t])
			next_state = state.add_input(x_t)
			y_star = np.reshape(dy.softmax(W_y * next_state.output() + b_y).npvalue(), -1)
			for nextWordID in np.argpartition(-y_star, n)[:n]:
				currentWord = self.tgt_id_to_token[nextWordID]
				if currentWord == unkSymbol:
					currentWord = self.src_id_to_token[unkIndex]
				currentLogProb = logProb + np.log(y_star[nextWordID])
				newWords = words + [currentWord]
				yield currentLogProb, generate_top_n(currentLogProb, next_state,newWords, nextWordID, n), newWords
		
		beamSize = 10
		trans = []
		currentBeam = [(0, generate_top_n(0, init_state, [startSymbol], self.tgt_token_to_id[startSymbol], beamSize), [startSymbol])]
		remainStep = self.max_len + 2
		while not trans and remainStep > 0:
			nextBeam = []
			while currentBeam:
				_, maxProbStep, _ = heappop(currentBeam)
				for next in maxProbStep:
					if isinstance(next[1], GeneratorType):
						heappush(nextBeam, next)
					else:
						trans.append(next)
			while len(nextBeam) > beamSize:
				heappop(nextBeam)
			currentBeam = nextBeam

		if trans:
			trans_sentence = max(trans)[-1][1:-1]
		else:
			trans_sentence = max(currentBeam)[-1][1:]
		
		return ' '.join(trans_sentence)

def main():
	model = dy.Model()

	trainer = dy.SimpleSGDTrainer(model)
	#training_src = read_file(sys.argv[1])
	#training_tgt = read_file(sys.argv[2])
	#trainFileName_src = "train.en-de.low.filt.de"
	#trainFileName_tgt = "train.en-de.low.filt.en"
	trainFileName_src = sys.argv[1]
	trainFileName_tgt = sys.argv[2]
	training_src = []
	training_tgt = []
	for line in open(trainFileName_src,'r'):
		fields = line.strip().split(' ')
		training_src.append(fields)
	for line in open(trainFileName_tgt,'r'):
		fields = line.strip().split(' ')
		training_tgt.append(fields)
		
	attention = Attention(model, training_src, training_tgt)
	
	(attention.src_lookup, attention.tgt_lookup, attention.l2r_builder, attention.r2l_builder, attention.dec_builder, attention.W_y, attention.b_y, attention.W1_att_f, attention.W1_att_e, attention.w2_att) = model.load('myModel')
	attention.l2r_builder.disable_dropout()
	attention.r2l_builder.disable_dropout()
	attention.dec_builder.disable_dropout()
	testFileName = sys.argv[3]
	testSent = []

	for line in open(testFileName,'r'):
		fields = line.strip().split(' ')
		testSent.append(fields)
	
	rst = []
	#translate
	ccc = 0
	for testS in testSent:
		ccc+=1
		rst.append(attention.translate_sentence(testS))
		
	f = open('rst','w')
	for i in rst:
		f.write(i+'\n')
	f.close()
	
	
	


if __name__ == '__main__': main()