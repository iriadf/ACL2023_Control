# %%
import sys
import numpy as np
from tqdm import tqdm
from minicons import scorer
import numpy as np
import pandas as pd
	
def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

# %%
def run_experiment(in_file_name, out_file_name, model_name):
	model = scorer.MaskedLMScorer(model_name = model_name , device = 'cpu')
	# %%
	in_df = pd.read_csv(in_file_name, header = 0, sep=';')
	n_sentences = in_df.shape[0]
	in_df.head()
	# %%
	adj_tokenization_len = np.zeros(n_sentences)
	adj_surprisal = np.zeros(n_sentences)
	sentence_surprisal = np.zeros(n_sentences)

	# %%
	for i in tqdm(range(n_sentences)):
		adj = in_df.ADJ[i]
		try: 
			adj_tokenization = model.token_score(' '+adj, surprisal = True)[0]
			adj_surprisal_in_context = model.token_score(in_df.sentence[i], surprisal = True)[0]
			_ , adj_idx = find_sub_list([x[0] for x in adj_tokenization],[x[0] for x in adj_surprisal_in_context])
			adj_tokenization_len[i] = len(adj_tokenization)
			adj_surprisal[i] = adj_surprisal_in_context[adj_idx][1]
			sentence_surprisal[i] = model.sequence_score(in_df.sentence[i], reduction = lambda x: -x.sum(0).item())[0]
		except  Exception as e :
			print(e)
			adj_tokenization_len[i] = - 1111111
			adj_surprisal[i] = - 1111111
			sentence_surprisal[i] = - 1111111
			pass
	# %%
	in_df['adj_tokenization_len'] = adj_tokenization_len.tolist()
	in_df['adj_surprisal'] = adj_surprisal.tolist()
	in_df['sentence_surprisal'] = sentence_surprisal.tolist()
	# %%
	in_df.to_csv(out_file_name, index = False, sep = ';')
	# %%

if __name__ == '__main__':
	
	# Input file
	in_file_name = sys.argv[1]

	# Models
	model_names = [ 
		'PlanTL-GOB-ES/roberta-large-bne', # RoBERTa-large
		'PlanTL-GOB-ES/roberta-base-bne', # RoBERTa-base
		'dccuchile/bert-base-spanish-wwm-cased', # BETO
		'bert-base-multilingual-cased', # mBERT
		'xlm-roberta-base', # XLM-RoBERTa-base
		'xlm-roberta-large', # XLM-RoBERTa-large
		] 

	out_file_names = [ 
		'output_acceptabilidade_roberta-large-bne.csv', # RoBERTa-base
		'output_acceptabilidade_roberta-base-bne.csv', # RoBERTa-base
		'output_acceptabilidade_bert-base-spanish-wwm-cased.csv', # BETO
		'output_acceptabilidade_bert-base-multilingual-cased.csv', # mBERT
		'output_acceptabilidade_xlm-roberta-base.csv', # XLM-RoBERTa-base
		'output_acceptabilidade_xlm-roberta-large.csv', # XLM-RoBERTa-large
		] 

	for model_name, out_file_name in zip (model_names, out_file_names):
		print("\n============================================================================\n")
		print("Computing acceptability values for the model: " + model_name)
		run_experiment(in_file_name=in_file_name, out_file_name=out_file_name, model_name=model_name)
		print("Results saved in: " + out_file_name)
	print("\n============================================================================\n")
