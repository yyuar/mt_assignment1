from collections import defaultdict
import dynet as dy
import numpy as np
import random
import sys

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
	def __attention_mlp(self, h_fs_matrix, h_e, isTrain):
		W1_att_f = dy.parameter(self.W1_att_f)
		W1_att_e = dy.parameter(self.W1_att_e)
		w2_att = dy.parameter(self.w2_att)
		# Calculate the alignment score vector
		# Hint: Can we make this more efficient?
		a_t = dy.transpose(dy.tanh(dy.colwise_add(W1_att_f * h_fs_matrix, W1_att_e * h_e))) * w2_att
		alignment = dy.softmax(a_t)
		c_t = h_fs_matrix * alignment
		if isTrain:
			return (c_t,None)
		#find the word with highest attention
		probs = alignment.vec_value()
		highAtteID = np.argmax(probs)
		return (c_t,highAtteID)

	# Training step over a single sentence pair
	def step(self, instances, enable_dropout=True):
		dy.renew_cg()
		
		if enable_dropout:
			self.l2r_builder.set_dropout(0.5)
			self.r2l_builder.set_dropout(0.5)
			self.dec_builder.set_dropout(0.5)
		else:
			self.l2r_builder.disable_dropout()
			self.r2l_builder.disable_dropout()
			self.dec_builder.disable_dropout()
		
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
			thisTgt = [startSymbol] + item[1] + [endSymbol for i in range(padNum+1)]
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
		start = dy.concatenate([dy.lookup_batch(self.tgt_lookup, [self.tgt_token_to_id['</S>'] for i in tgt_sents]), c_t])
		dec_state = self.dec_builder.initial_state().add_input(start)
		#loss = dy.pickneglogsoftmax_batch(W_y * dec_state.output() + b_y,[self.tgt_token_to_id[tgt_sent[0]] for tgt_sent in tgt_sents])
		#losses.append(loss)
		
		for i in range(tgtSenLen):
			#cw : item[i] nw:item[i+1]
			h_e = dec_state.output()
			c_t = self.__attention_mlp(h_fs_matrix, h_e,1)[0]
			# Get the embedding for the current target word
			embed_t = dy.lookup_batch(self.tgt_lookup, [self.tgt_token_to_id[tgt_sent[i]] for tgt_sent in tgt_sents])
			# Create input vector to the decoder
			x_t = dy.concatenate([embed_t, c_t])
			dec_state = dec_state.add_input(x_t)
			o_en = dec_state.output()
			if enable_dropout:
				o_en = dy.dropout(o_en, 0.5)
			loss = dy.pickneglogsoftmax_batch(W_y * o_en + b_y,[self.tgt_token_to_id[tgt_sent[i+1]] for tgt_sent in tgt_sents])
			thisMask = dy.inputVector(masks[i])
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
		trans_sentence = [startSymbol]
		cw = trans_sentence[-1]
		#initial context
		c_t = dy.vecInput(self.hidden_size * 2)
		start = dy.concatenate([dy.lookup(self.tgt_lookup, self.tgt_token_to_id[endSymbol]), c_t])
		dec_state = self.dec_builder.initial_state().add_input(start)
		while len(trans_sentence) < self.max_len:
			h_e = dec_state.output()
			getAttention = self.__attention_mlp(h_fs_matrix, h_e,0)
			c_t = getAttention[0]
			embed_t = dy.lookup(self.tgt_lookup, self.tgt_token_to_id[cw])
			x_t = dy.concatenate([embed_t, c_t])
			dec_state = dec_state.add_input(x_t)
			y_star = dy.softmax(W_y * dec_state.output() + b_y).vec_value()
			next_wordID = np.argmax(y_star)
			cw = self.tgt_id_to_token[next_wordID]
			cpcw = cw  #store the original word for computing next word
			if cw==unkSymbol:
				#find the source word with highest attention score
				keyWord = sent[getAttention[1]]
				if self.src_token_to_id[keyWord] == self.src_token_to_id[unkSymbol]:
					cw = keyWord #special word . simply pass it source word out
				else:
					#find the target word with second max prob
					#prob: y_star
					next_wordID = np.argpartition(y_star,1)[1]
					cw = self.tgt_id_to_token[next_wordID]
			if cw == endSymbol:
				break
			if cw != startSymbol:
				trans_sentence.append(cw)
			cw = cpcw #get the original cw

		return ' '.join(trans_sentence[1:])

def main():
	model = dy.Model()
	trainer = dy.AdamTrainer(model)
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
	trainingSet = zip(training_src,training_tgt)
	dic = defaultdict(list)
	for item in trainingSet:
		dic[len(item[0])].append(item)
		
	#validation
	devFileName_src = sys.argv[3]
	devFileName_tgt = sys.argv[4]
	dev_src = []
	dev_tgt = []
	for line in open(devFileName_src,'r'):
		fields = line.strip().split(' ')
		dev_src.append(fields)
	for line in open(devFileName_tgt,'r'):
		fields = line.strip().split(' ')
		dev_tgt.append(fields)
	devSet = zip(dev_src,dev_tgt)
	dicDev = defaultdict(list)
	for item in devSet:
		dicDev[len(item[0])].append(item)
	
	attention = Attention(model, training_src, training_tgt)
	
	(attention.src_lookup, attention.tgt_lookup, attention.l2r_builder, attention.r2l_builder, attention.dec_builder, attention.W_y, attention.b_y, attention.W1_att_f, attention.W1_att_e, attention.w2_att) = model.load('myModel')
	
	epoch = 20
	batchSize = 32
	minDevLoss = float('inf')
	for iter in range(epoch):
		totalLoss = 0
		totalNum = 0
		print "start epoch: " + str(iter)
		for key, sentSameLenList in dic.items():
			#sentSameLenList
			index = 0
			listLen = len(sentSameLenList)
			#shuffle!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			random.shuffle(sentSameLenList)
			while index<listLen:
				loss, wordNum = attention.step(sentSameLenList[index:min(index+batchSize,listLen)])
				index += batchSize
				totalLoss += loss.value()
				totalNum += wordNum
				loss.backward()
				trainer.update()
			#print 'complete: ' + str(listLen) + ' instances'
		print "epoch" + str(iter) + ': trainingLoss ' + str(totalLoss/totalNum)
		
		#test the loss on validation set
		devTtlLoss = 0
		devTtlNum = 0
		for key, sentSameLenList in dicDev.items():
			index = 0
			listLen = len(sentSameLenList)
			while index<listLen:
				loss, wordNum = attention.step(sentSameLenList[index:min(index+batchSize,listLen)], False)
				index += batchSize
				devTtlLoss += loss.value()
				devTtlNum += wordNum
		curDevLoss = devTtlLoss / devTtlNum
		print "epoch" + str(iter) + ': devLoss ' + str(curDevLoss)
		if curDevLoss<minDevLoss:
			model.save('myModel',[attention.src_lookup, attention.tgt_lookup, attention.l2r_builder, attention.r2l_builder, attention.dec_builder, attention.W_y, attention.b_y, attention.W1_att_f, attention.W1_att_e, attention.w2_att])
			minDevLoss = curDevLoss
	
	#devFileName_src = sys.argv[3]
	#devFileName_tgt = sys.argv[4]
	testFileName = sys.argv[5]
	testSent = []

	for line in open(testFileName,'r'):
		fields = line.strip().split(' ')
		testSent.append(fields)
	
	rst = []
	#translate
	for testS in testSent:
		rst.append(attention.translate_sentence(testS))
		
	f = open('rst','w')
	for i in rst:
		f.write(i+'\n')
	f.close()
	
	
	


if __name__ == '__main__': main()