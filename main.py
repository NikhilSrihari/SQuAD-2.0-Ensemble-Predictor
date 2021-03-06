import pandas as pd
import numpy as np
import json
from EnsembleClassifier import EnsembleClassifier


devfile_location = 'SQuAD/SQuAD-data/dev-v2.0.json'
B_opfile_location = './SQuAD/SQuAD-explorer/models/v2.0/BERT (single model) (Google AI Language).json'
E_opfile_location = './SQuAD/SQuAD-explorer/models/v2.0/BiDAF + Self Attention + ELMo (single model) (Allen Institute for Artificial Intelligence [modified by Stanford]).json'
N_opfile_location = './SQuAD/SQuAD-explorer/models/v2.0/nlnet (single model) (Microsoft Research Asia).json'


def fetchDevData():
	with open(B_opfile_location) as json_data:
	    return json.load(json_data)


def fetchModelsOP():
	with open(B_opfile_location) as json_data:
	    B_op = json.load(json_data)
	with open(E_opfile_location) as json_data:
	    E_op = json.load(json_data)
	with open(N_opfile_location) as json_data:
	    N_op = json.load(json_data)
	return B_op, E_op, N_op


def writeJSON(data, fileName):
	with open(fileName+'.json', 'w') as outfile:
		json.dump(data, outfile)


def main():
	data = fetchDevData()
	B_op, E_op, N_op = fetchModelsOP()
	print("Starting Ensemble Classifier: ")
	ec = EnsembleClassifier("MAX_VOTING")
	final_op1, final_voter1 = ec.predict1(B_op, E_op, N_op)
	writeJSON(final_op1, "final_predictions1")
	writeJSON(final_voter1, "final_predictionsVoter1")
	print("Predictions written to file final_predictions1.json")
	final_op2, final_voter2 = ec.predict2(B_op, E_op, N_op)
	writeJSON(final_op2, "final_predictions2")
	writeJSON(final_voter2, "final_predictionsVoter2")
	print("Predictions written to file final_predictions2.json")


main()