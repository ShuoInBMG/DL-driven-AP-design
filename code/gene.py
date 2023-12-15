import random
import numpy as np
import torch
import os
from tqdm.auto import tqdm
from .models import TimeSeriesTransformer

class evolution:
    def __init__(self, 
                 select_rate,           # how many left
                 temp_file_path,        # write history fasta
                 max_length,            # length of aa
                 score_model):
        self.aa_table = self._init_aa_table()
        self.aa_table_r = self._init_aa_table_r()
        self.select_rate = select_rate
        # temp_file_path is a empty folder.
        self.temp_file_path = temp_file_path
        self.max_length = max_length
        self.score_model = score_model
    def _init_aa_table(self):
        aa_table = {}
        aa_table["A"] = 0
        aa_table["C"] = 1
        aa_table["D"] = 2
        aa_table["E"] = 3
        aa_table["F"] = 4
        aa_table["G"] = 5
        aa_table["H"] = 6
        aa_table["I"] = 7
        aa_table["K"] = 8
        aa_table["L"] = 9
        aa_table["M"] = 10
        aa_table["N"] = 11
        aa_table["P"] = 12
        aa_table["Q"] = 13
        aa_table["R"] = 14
        aa_table["S"] = 15
        aa_table["T"] = 16
        aa_table["V"] = 17
        aa_table["W"] = 18
        aa_table["Y"] = 19
        return aa_table
    def _init_aa_table_r(self):
        aa_table = {}
        aa_table[0] = "A"
        aa_table[1] = "C"
        aa_table[2] = "D"
        aa_table[3] = "E"
        aa_table[4] = "F"
        aa_table[5] = "G"
        aa_table[6] = "H"
        aa_table[7] = "I"
        aa_table[8] = "K"
        aa_table[9] = "L"
        aa_table[10] = "M"
        aa_table[11] = "N"
        aa_table[12] = "P"
        aa_table[13] = "Q"
        aa_table[14] = "R"
        aa_table[15] = "S"
        aa_table[16] = "T"
        aa_table[17] = "V"
        aa_table[18] = "W"
        aa_table[19] = "Y"
        return aa_table
    def _check_empty_folder(self):
        temp_folder_exist = os.path.exists(self.temp_file_path)
        if not temp_folder_exist:
            os.makedirs(self.temp_file_path)
            print("INFO: Folder not available, a new one was made.")
        else:
            print("INFO: Temp folder is ready.")
        fasta_path = os.path.join(self.temp_file_path,
                                  "fasta")
        if not os.path.exists(fasta_path):
            os.makedirs(fasta_path)
        history_path = os.path.join(self.temp_file_path,
                                    "history")
        if not os.path.exists(history_path):
            os.makedirs(history_path)
    def _write_into_temp(self, sequence_list:list, epoch):
        # Make a new .fasta file under the temp_folder
        sequence_set = set(sequence_list)
        sequence_list = list(sequence_set)
        generation_path = self.temp_file_path + f"\{epoch}.fasta"
        sequence_choices = random.sample(sequence_list, 20)
        with open(generation_path, "w") as f:
            for i,seqeunce in enumerate(sequence_choices):
                f.write(f">{i}\n")
                f.write(f"{seqeunce}\n")
            f.close()
    def selection(self, population, fitness):
        total_fitness = np.sum(np.array(fitness))
        probabilities = [f / total_fitness for f in fitness]
        selected = random.choices(population = population,
                                  weights = probabilities,
                                  k = self.select_rate)
        return selected
    def crossover(self, str1, str2):
        if self.whether_constant_length:
            point1 = random.randint(1, (len(str1))-1)
            offspring1 = str1[:point1] + str2[point1:]
            offspring2 = str2[:point1] + str1[point1:]
        else:
            point1 = random.randint(1, (len(str1))-1)
            point2 = random.randint(1, (len(str2))-1)
            offspring1 = str1[:point1] + str2[point2:]
            offspring2 = str2[:point2] + str1[point1:]
        return offspring1, offspring2
    def mutation(self, seq, mutation_rate):
        new_seq = list(seq)
        for i in range(len(seq)):
            if np.random.random() < mutation_rate:
                new_index = np.random.randint(0, 20)
                new_seq[i] = self.aa_table_r[new_index]
            else:
                pass
        return "".join(new_seq)
    def replace(self, sequence, offspring):
        for raw_seq in sequence:
            offspring.append(raw_seq)
        return offspring
    def init_sequences(self, population):
        random_matrix = np.random.randint(0, 20, size=(population, self.max_length))
        return random_matrix
    def pre_transform(self, sequence_population_list):
        return_embedding = []
        for seq in sequence_population_list:
            index_seq = [self.aa_table[x] for x in seq]
            return_embedding.append(index_seq)
        return_embedding = np.array(return_embedding)
        return return_embedding
    def translate(self, population):
        return_string = []
        for seq in population:
            symbol_sequence = [self.aa_table_r[x] for x in seq]
            symbol_string = "".join(symbol_sequence)
            return_string.append(symbol_string)
        return return_string
    def genetic(self, 
                mutation_rate,
                num_generation,
                population,
                constant_length,
                temp_interval):
        self.whether_constant_length = constant_length
        sequence_population_list = self.init_sequences(population)
        sequence_population_list = self.translate(sequence_population_list)
        self.start_record = sequence_population_list
        genetic_history = []
        self._check_empty_folder()
        loop = tqdm(total=num_generation)
        for generation in range(num_generation):
            # Translate sequence to torch.Tensor
            if temp_interval == False:
                pass
            elif generation % temp_interval == 0:
                self._write_into_temp(sequence_population_list,generation)
            population_seq = self.pre_transform(sequence_population_list) # torch.long
            # Predict fitness (AP value) by torch neural network model
            fitness = self.score_model.predict(population_seq)
            # Save history
            genetic_history.append([np.mean(fitness), np.std(fitness)])
            # Select according to weight
            selected_sequence = self.selection(sequence_population_list, fitness)
            # Cross
            offspring_sequence = []
            for i in range(0, len(selected_sequence), 2):
                offspring_sequence.extend(self.crossover(selected_sequence[i], selected_sequence[i+1]))
            # Mutation
            if mutation_rate > 0:
                for i in range(len(offspring_sequence)):
                    offspring_sequence[i] = self.mutation(offspring_sequence[i], mutation_rate)
            sequence_population_list = self.replace(selected_sequence, offspring_sequence)
            random.shuffle(sequence_population_list)
            loop.update(1)
        loop.close()

        return sequence_population_list, np.array(genetic_history)

class score_model:
    def __init__(self, model_path):
        self.model = TimeSeriesTransformer(input_size=21, output_size=1)
        self.model.load_state_dict(torch.load(model_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def predict(self, sequence):
        idx_sequence = sequence + 1
        num, length = idx_sequence.shape
        padding = length - 10
        if padding > 0:
            padding_array = np.zeros(shape = (num, padding))
            idx_sequence = np.concatenate([idx_sequence, padding_array], axis = 1)
        input_sequence = torch.from_numpy(idx_sequence).long()
        input_sequence = input_sequence.to(self.device)

        mask = torch.tril(torch.ones(1, 1)).to(self.device) == 0
        AP_value = self.model(input_sequence, mask)
        return AP_value.cpu().numpy()