from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import numpy as np
import sys
import csv
import pandas as pd
from tqdm import tqdm

def csv_to_dict(dict_file):
	vocab = {}
	with open(dict_file) as f:
		reader = csv.reader(f, delimiter=';')
		for row in reader:
			vocab[row[0]] = row[1]
	return vocab



def run_experiment(in_file_name, out_file_name, model_name, dict_file, N):

	tokenizer=AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
	model = AutoModelWithLMHead.from_pretrained(model_name)
	vocab = csv_to_dict(dict_file)

	adjs_in_vocab_keys = []
	adjs_tokenization_len = []
	n_masc=[];prob_masc=[]
	n_fem=[];prob_fem=[]
	n_neutral=[];prob_neutral=[]
	n_nonsense=[];prob_nonsense=[]
	success_prob_topN = []
	verb_in_vocab_keys=[]
	verb_tokenization_len = []

	in_df = pd.read_csv(in_file_name, header = 0, sep=';')

	n=in_df.shape[0]
	ind=np.arange(n)

	if len(tokenizer.encode('*'))!=3:
		print('ERROR')
		sys.exit()

	mask_data_set=tokenizer.encode('*',return_tensors="pt")[0][1]
	resultados=[];score_correct=[];score_wrong=[]
	pred_1=[]; pred_2=[]; pred_3=[]; pred_4=[]; pred_5=[]
	prob_1=[]; prob_2=[];  prob_3=[]; prob_4=[];  prob_5=[]

	for i in tqdm(ind):
		if (not in_df.correct_adj[i] in tokenizer.vocab.keys()) or (not in_df.incorrect_adj[i] in tokenizer.vocab.keys()):
			adjs_in_vocab_keys.append(str(0))
			if tokenizer.tokenize(in_df.correct_adj[i])[:-1]==tokenizer.tokenize(in_df.incorrect_adj[i])[:-1]:
				aux = "".join(tokenizer.tokenize(in_df.correct_adj[i])[:-1])
				if len(aux)>0 and not aux[0].islower(): aux = aux[1:]
				sequence = in_df.sentence[i].replace('*',f'{aux}{tokenizer.mask_token}')
				input_ids = tokenizer.encode(sequence, return_tensors="pt")
				mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
				token_logits = model(input_ids)[0] 
				mask_token_logits = token_logits[0, mask_token_index, :]
				mask_token_logits = torch.softmax(mask_token_logits, dim=1)

				sought_after_token_id = tokenizer.encode(in_df.correct_adj[i], add_special_tokens=False)[-1]
				sought_after_token_id2 = tokenizer.encode(in_df.incorrect_adj[i], add_special_tokens=False)[-1]

				score_correct_i = mask_token_logits[:, sought_after_token_id].detach().item()

				score_wrong_i = mask_token_logits[:, sought_after_token_id2].detach().item()
				score_correct.append(str(score_correct_i))
				score_wrong.append(str(score_wrong_i))
				if score_correct_i>score_wrong_i:
					resultados.append(str(1))
				elif score_correct_i<=score_wrong_i:
					resultados.append(str(0))
				written = True
				adjs_tokenization_len.append(str(len(tokenizer.tokenize(in_df.correct_adj[i]))))
			else:
				adjs_tokenization_len.append(str(-1))
				score_correct.append(str(-1))
				score_wrong.append(str(-1))
				resultados.append(str(-1))
				written = True
		else:
			adjs_in_vocab_keys.append(str(0))
			adjs_tokenization_len.append(str(1))
			written = False

		sequence=in_df.sentence[i]
		ids_sequence=tokenizer.encode(sequence,return_tensors="pt")
		mask_ind=torch.where(ids_sequence[0]==mask_data_set)[0]
		ids_sequence[0][mask_ind]=tokenizer.mask_token_id

		logits=model(ids_sequence).logits
		mask_logits=logits[0][mask_ind]
		mask_logits=torch.softmax(mask_logits,dim=1)

		if not written:
			ids_correct=tokenizer.encode(in_df.correct_adj[i],return_tensors="pt")[0][1]
			score_correct_i=mask_logits[0][ids_correct]
			score_correct_i=score_correct_i.detach().item()
			ids_wrong=tokenizer.encode(in_df.incorrect_adj[i],return_tensors="pt")[0][1]
			score_wrong_i=mask_logits[0][ids_wrong]
			score_wrong_i=score_wrong_i.detach().item()
			score_correct.append(str(score_correct_i))
			score_wrong.append(str(score_wrong_i))
			if score_correct_i>score_wrong_i:
				resultados.append(str(1))
			elif score_correct_i<=score_wrong_i:
				resultados.append(str(0))
		
		top_5 = torch.topk(mask_logits, 5, dim=1) 
		pred_1.append(str(tokenizer.decode([top_5.indices[0].tolist()[0]])))
		pred_2.append(str(tokenizer.decode([top_5.indices[0].tolist()[1]])))
		pred_3.append(str(tokenizer.decode([top_5.indices[0].tolist()[2]])))
		pred_4.append(str(tokenizer.decode([top_5.indices[0].tolist()[3]])))
		pred_5.append(str(tokenizer.decode([top_5.indices[0].tolist()[4]])))
		prob_1.append(str(top_5.values[0].tolist()[0]))
		prob_2.append(str(top_5.values[0].tolist()[1]))
		prob_3.append(str(top_5.values[0].tolist()[2]))
		prob_4.append(str(top_5.values[0].tolist()[3]))
		prob_5.append(str(top_5.values[0].tolist()[4]))


		top_N = torch.topk(mask_logits, N, dim=1)
		top_N_tokens = zip(top_N.indices[0].tolist(), top_N.values[0].tolist())
		
		n_masc_i=0; prob_masc_i=0; n_fem_i=0; prob_fem_i=0;
		n_neutral_i=0; prob_neutral_i=0; n_nonsense_i=0; prob_nonsense_i=0
		x=0
		for token, score in top_N_tokens:
			word = tokenizer.decode([token])
			word = word.replace(" ","")
			if word in vocab:
				if vocab[word] == 'M':
					n_masc_i = n_masc_i + 1
					prob_masc_i = prob_masc_i + score
				elif vocab[word] == 'F':
					n_fem_i = n_fem_i + 1
					prob_fem_i = prob_fem_i + score
				elif vocab[word] == 'C':
					n_neutral_i = n_neutral_i + 1
					prob_neutral_i = prob_neutral_i + score
			else:
				n_nonsense_i = n_nonsense_i + 1
				prob_nonsense_i = prob_nonsense_i + score

		n_masc.append(n_masc_i)
		prob_masc.append(prob_masc_i)
		n_fem.append(n_fem_i)
		prob_fem.append(prob_fem_i)
		n_neutral.append(n_neutral_i)
		prob_neutral.append(prob_neutral_i)
		n_nonsense.append(n_nonsense_i)
		prob_nonsense.append(prob_nonsense_i)

		if in_df.correct_gender[i] == 'masc':
			if prob_masc_i > prob_fem_i: 
				success_prob_topN.append(str(1))
			else:
				success_prob_topN.append(str(0))
		else :
			if prob_fem_i > prob_masc_i:
				success_prob_topN.append(str(1))
			else:
				success_prob_topN.append(str(0))
		
		verb = in_df.control_verb[i]

		if verb in tokenizer.vocab.keys():
			verb_in_vocab_keys.append(str(1))
		else:
			#if the verb isnt in the model vocab
			verb_in_vocab_keys.append(str(0))
		
		verb_tokenization_len.append(str(len(tokenizer.tokenize(verb))))


	print('Writing results to file')

	in_df['adjs_in_vocab_keys'] = adjs_in_vocab_keys
	in_df['adjs_tokenization_len'] = adjs_tokenization_len
	in_df['prob_c'] = score_correct
	in_df['prob_w'] = score_wrong
	in_df['success'] = resultados
	in_df['n_masc'] = n_masc
	in_df['prob_masc'] = prob_masc
	in_df['n_fem'] = n_fem
	in_df['prob_fem'] = prob_fem
	in_df['n_neutral'] = n_neutral
	in_df['prob_neutral'] = prob_neutral
	in_df['n_nonsense'] = n_nonsense
	in_df['prob_nonsense'] = prob_nonsense
	in_df['success_prob_topN'] = success_prob_topN

	in_df['pred_1'] = pred_1
	in_df['pred_2'] = pred_2
	in_df['pred_3'] = pred_3
	in_df['pred_4'] = pred_4
	in_df['pred_5'] = pred_5

	in_df['prob_1'] = prob_1
	in_df['prob_2'] = prob_2
	in_df['prob_3'] = prob_3
	in_df['prob_4'] = prob_4
	in_df['prob_5'] = prob_5

	in_df['verb_in_vocab_keys'] = verb_in_vocab_keys
	in_df['verb_tokenization_len'] = verb_tokenization_len


	in_df.to_csv(out_file_name, index = False, sep = ';')

if __name__ == '__main__':
	
	# Input
	in_file_name = sys.argv[1]

	# Models
	model_names = [ 
		'dvilares/bertinho-gl-base-cased', # Bertinho-base
		'dvilares/bertinho-gl-small-cased', # Bertinho-small
		'marcosgg/bert-base-gl-cased', # BERT base
		'marcosgg/bert-small-gl-cased', # BERT small
		'bert-base-multilingual-cased', # mBERT
		'xlm-roberta-base', # XLM-RoBERTa-base
		'xlm-roberta-large', # XLM-RoBERTa-large
		] 

	out_file_names = [
		'output_predicion_bertinho-base.csv', # Bertinho-base
		'output_predicion_bertinho-small.csv', # Bertinho-small
		'output_predicion_bert-base.csv', # BERT base
		'output_predicion_bert-small.csv', # BERT small
		'output_predicion_bert-base-multilingual-cased.csv', # mBERT
		'output_predicion_xlm-roberta-base.csv', # XLM-RoBERTa-base
		'output_predicion_xlm-roberta-large.csv', # XLM-RoBERTa-large
		]

	for model_name, out_file_name in zip (model_names, out_file_names):
		print("\n============================================================================\n")
		print("Running experiment with model: " + model_name)
		run_experiment(in_file_name=in_file_name, out_file_name=out_file_name, model_name=model_name, dict_file = 'adjectives_galician.txt', N = 100)
		print("Results saved in: " + out_file_name)
	print("\n============================================================================\n")
