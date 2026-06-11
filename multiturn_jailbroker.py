import requests
import pandas as pd
import datetime
import os
import traceback
import re
import importlib
import llm_aggregator
importlib.reload(llm_aggregator)
import my_model
importlib.reload(my_model)
from llm_aggregator import LLMAggregator
from my_model import MyModel


"""   
INFO ABOUT THIS SOURCE CODE

This source code provides functions to launch single-turn and multi-turn jailbreaking attacks against open-source Large Language Models (LLMs).
Below are the main functions included:

1. launch_single_turn_attack(): Launches a single-turn attack against a targeted LLM (open or close), with optional past-tense reformulation.
2. launch_multiple_turn_attack(): Launches a multi-turn attack against a targeted LLM (open or close), with optional past-tense reformulation.

"""


class MultiTurnJailbroker(MyModel):

    # Function of class initialization (generator)
    # By default the 1st LLM from opensource LLMs and the 1st temperature are selected from the lists
    def __init__(self, is_selected_llm_open=True, selected_llm_index=0, selected_temp_index=0):
        # We invoke the super generator from the super class
        super().__init__(is_selected_llm_open, selected_llm_index, selected_temp_index)

    # Method to update the LLM model index and model name.
    # By default the chosen LLM is opensource and the first
    def set_llm_index(self, is_selected_llm_open=True, newIndex: int = 0):
        # We invoke the super function from the super class
        super().set_llm_index(is_selected_llm_open, newIndex)

    
    # Method to update the temperature index and value.
    def set_temperature_index(self, newIndex: int):
        # We invoke the super function from the super class
        super().set_temperature_index(newIndex)



    # Function to launch single turn attack against a targeted LLM with past tense reformulation or not
    # min_row and max_row determine the range of rows to select in the table of input file for optimization of the attack process
    # Input:
    # is_past: Indicate whether we're using past tense reformulation or not
    # min_row: Starting index of the rows to select from the input file
    # max_row: Ending index of the rows to select from the input file
    # Output:
    # This function does not return any value but will create a new file, named according to the output_file variable, which will contain the LLM response for the single-turn attacks
    def launch_single_turn_attack(self, is_past=False, min_row=None, max_row=None):
        try:
            # We use a naming convention for the output file with the following syntax:
            # <LLM_model_name>_attack_N_<Nb of interation>_<present or past>.csv
            if is_past:
                output_file = LLMAggregator.TEST_FOLDER + '' + self.get_model_name() + '/' + self.get_model_name() + '_attack_N_1_past.csv'
            else:
                output_file = LLMAggregator.TEST_FOLDER + '' + self.get_model_name() + '/' + self.get_model_name() + '_attack_N_1_present.csv'
            
            # Read the input CSV file
            df = pd.read_csv(LLMAggregator.INPUT_FILE)
        
            # Check if 'Prompt' column exists
            if 'Prompt' not in df.columns:
                print("The 'Prompt' column is missing in the input file.")
                return
                    
            # Check if 'SemanticCategory' column exists
            if 'SemanticCategory' not in df.columns:
                print("The 'SemanticCategory' column is missing in the input file.")
                return
        
            # Check if 'SubTopic' column exists
            if 'SubTopic' not in df.columns:
                print("The 'SubTopic' column is missing in the input file.")
                return
    
            # Set defaults if min_row or max_row is None
            if min_row is None:
                min_row = 1
            if max_row is None:
                max_row = len(df)
    
            if min_row >= max_row:
                print(f"Invalid range: min_row ({min_row}) must be less than max_row ({max_row}).")
                return
    
            # Filter rows based on the range
            df = df.iloc[(min_row-1):max_row]
            
            # Determine starting Unique ID based on existing output file
            starting_id = 1
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                if 'Unique ID' in existing_df.columns:
                    starting_id = existing_df['Unique ID'].max() + 1
    
    
            # Initialize an empty list to hold the new rows
            new_rows = []
    
            # Index counter
            count = 0
            
            # Iterate over each row in pd
            for index, row in df.iterrows():
                # Initialize an empty conversation
                conversations = []
                # new row ID
                Row_ID = starting_id + count
    
                # For each row in the input file, we prepare a prompt for the LLM and retrieve the response that we'll put in the output file
                # If single past tense reformulation prompt, we select 'Past' column from the input file
                if is_past:
                    rsp = LLMAggregator.process_conversation(self.is_model_open_source(), self.get_llm_index(), self.get_temp(), conversations, row['Past'])
                    d = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # We create a new row in the output file with the prompt request (in past) and the generated response from the LLM along with other useful info.
                    new_rows.append({'Prompt': row['Past'],
                                 'SemanticCategory': row["SemanticCategory"], 
                                 'SubTopic': row['SubTopic'], 
                                 'QueryType': row['QueryType'],
                                 'Model': self.get_llm(),              
                                 'Version': self.get_llm_version(),
                                 'Temp.': self.get_temp(),    
                                 'Unique ID': Row_ID, 
                                 'Multi Step': 0, 
                                 'Response': rsp,  
                                 'Date Time': d })
                else:
                    rsp = LLMAggregator.process_conversation(self.is_model_open_source(), self.get_llm_index(), self.get_temp(), conversations, row['Prompt'])
                    d = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # We create a new row in the output file with the prompt request (in present) and the generated response from the LLM along with other useful info.
                    new_rows.append({'Prompt': row['Prompt'],
                                 'SemanticCategory': row["SemanticCategory"], 
                                 'SubTopic': row['SubTopic'], 
                                 'QueryType': row['QueryType'],
                                 'Model': self.get_llm(),              
                                 'Version': self.get_llm_version(),
                                 'Temp.': self.get_temp(),
                                 'Unique ID': Row_ID, 
                                 'Multi Step': 0, 
                                 'Response': rsp,
                                 'Date Time': d })
                count = count + 1
            
            # Create the new DataFrame with new rows and columns
            output_df = pd.DataFrame(new_rows)
    
            # Make sure the directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
            # Check if the file already exists
            file_exists = os.path.exists(output_file)
    
            # Write the new DataFrame to the output CSV file in a 'append' mode while ignoring the first row (for column names)
            output_df.to_csv(output_file, mode='a', header=not file_exists, index=False)
            print(f"LLM responses written to {output_file}.")
    
        except FileNotFoundError:
            print(f"The file {LLMAggregator.INPUT_FILE} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Full traceback:")
            traceback.print_exc()  # This will print the detailed traceback



    # Function to launch multiple turn attack against a targeted LLM with past tense reformulation or not
    # min_row and max_row determine the range of rows to select in the table of input file for optimization of the attack process
    # Input:
    # nb_iterations: The number of iteration for the multi-turn attack
    # is_past: Indicate whether we're using past tense reformulation or not
    # min_row: Starting index of the rows to select from the input file
    # max_row: Ending index of the rows to select from the input file
    # Output:
    # This function does not return any value but will create a new file, named according to the output_file variable, which will contain the LLM response for the multi-turn attacks
    def launch_multiple_turn_attack(self, nb_iterations=2, is_past=False, min_row=None, max_row=None):
        try:
            # We use a naming convention for the output file with the following syntax:
            # <LLM_model_name>_attack_N_<Nb of interation>_<present or past>.csv
            if is_past:
                output_file = LLMAggregator.TEST_FOLDER + '' + self.get_model_name() + '/' + self.get_model_name() + '_attack_N_'+ str(nb_iterations) +'_past.csv'
            else:
                output_file = LLMAggregator.TEST_FOLDER + '' + self.get_model_name() + '/' + self.get_model_name() + '_attack_N_'+ str(nb_iterations) +'_present.csv'
        
            # Read the input CSV file which is the generated multi turn dataset corresponding to nb of iterations
            df = pd.read_csv(LLMAggregator.MULTITURN_DATASETS[""+str(nb_iterations)])
        
            # Check if 'Prompt' column exists
            if 'Prompt' not in df.columns:
                print("The 'Prompt' column is missing in the input file.")
                return
                    
            # Check if 'SemanticCategory' column exists
            if 'SemanticCategory' not in df.columns:
                print("The 'SemanticCategory' column is missing in the input file.")
                return
        
            # Check if 'SubTopic' column exists
            if 'SubTopic' not in df.columns:
                print("The 'SubTopic' column is missing in the input file.")
                return
    
            # Set defaults if min_row or max_row is None
            if min_row is None:
                min_row = 1
            if max_row is None:
                max_row = len(df)
    
            if min_row > max_row:
                print(f"Invalid range: min_row ({min_row}) must be less than max_row ({max_row}).")
                return
            
            # Filter rows based on the range
            df = df.iloc[(min_row-1):max_row]
            
            # Determine starting Unique ID based on existing output file
            starting_id = 1
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                if 'Unique ID' in existing_df.columns:
                    starting_id = existing_df['Unique ID'].max() + 1
    
            # Initialize an empty list to hold the new rows
            new_rows = []
    
            # Index counter
            count = 0
            
            # Iterate over each row in panda1
            for index, row in df.iterrows():
                
                # Initialize an empty conversation to keep the chat history
                conversations = []
                Row_ID = starting_id + count
    
    
                # If past tense reformulation is required
                if is_past:
                    # LLM query and response for the prompt of turn 1
                    rsp = LLMAggregator.process_conversation(self.is_model_open_source(), self.get_llm_index(), self.get_temp(), conversations, row['Past Turn 1'])
                    d = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # We create a new row in the output file with the prompt 1 request (in past) and the generated response from the LLM along with other useful info.
                    conversations.append({"role": "assistant", "content": rsp})
                    new_rows.append({'Prompt': row['Prompt'], 
                                     'New Prompt': row['Past Turn 1'], 
                                     'SemanticCategory': row["SemanticCategory"], 
                                     'SubTopic': row['SubTopic'], 
                                     'QueryType': row['QueryType'],
                                     'Model': self.get_llm(),              
                                     'Version': self.get_llm_version(),
                                     'Temp.': self.get_temp(),
                                     'Unique ID': Row_ID, 
                                     'Multi Step': 1, 
                                     'Response': rsp,
                                     'Date Time': d })
                    
                    # LLM query and response for the prompt of turn 2
                    rsp = LLMAggregator.process_conversation(self.is_model_open_source(), self.get_llm_index(), self.get_temp(), conversations, row['Past Turn 2'])
                    d = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # We create a new row in the output file with the prompt 2 request (in past) and the generated response from the LLM along with other useful info.
                    conversations.append({"role": "assistant", "content": rsp})
                    new_rows.append({'Prompt': row['Prompt'], 
                                     'New Prompt': row['Past Turn 2'], 
                                     'SemanticCategory': row["SemanticCategory"], 
                                     'SubTopic': row['SubTopic'],
                                     'QueryType': row['QueryType'],
                                     'Model': self.get_llm(),              
                                     'Version': self.get_llm_version(),
                                     'Temp.': self.get_temp(),
                                     'Unique ID': Row_ID, 
                                     'Multi Step': 2, 
                                     'Response': rsp,
                                     'Date Time': d })
                    
                    # If it's 3 turn attack, we continue to query the LLM and get the response for the prompt of turn 3 in past tense
                    if nb_iterations == 3:
                        rsp = LLMAggregator.process_conversation(self.is_model_open_source(), self.get_llm_index(), self.get_temp(), conversations, row['Past Turn 3'])
                        d = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # We create a new row in the output file with the prompt 3 request (in past) and the generated response from the LLM along with other useful info.
                        conversations.append({"role": "assistant", "content": rsp})
                        new_rows.append({'Prompt': row['Prompt'], 
                                     'New Prompt': row['Past Turn 3'], 
                                     'SemanticCategory': row["SemanticCategory"], 
                                     'SubTopic': row['SubTopic'],
                                     'QueryType': row['QueryType'],
                                     'Model': self.get_llm(),              
                                     'Version': self.get_llm_version(),
                                     'Temp.': self.get_temp(),
                                     'Unique ID': Row_ID, 
                                     'Multi Step': 3, 
                                     'Response': rsp,
                                     'Date Time': d })
                        
    
                # If past tense reformulation is not required    
                else:
                    # LLM query and response for the prompt of turn 1
                    rsp = LLMAggregator.process_conversation(self.is_model_open_source(), self.get_llm_index(), self.get_temp(), conversations, row['Present Turn 1'])
                    d = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    conversations.append({"role": "assistant", "content": rsp})
                    # We create a new row in the output file with the prompt 1 request (in present) and the generated response from the LLM along with other useful info.
                    new_rows.append({'Prompt': row['Prompt'],
                                     'New Prompt': row['Present Turn 1'],
                                     'SemanticCategory': row["SemanticCategory"], 
                                     'SubTopic': row['SubTopic'],
                                     'QueryType': row['QueryType'],
                                     'Model': self.get_llm(),              
                                     'Version': self.get_llm_version(),
                                     'Temp.': self.get_temp(),
                                     'Unique ID': Row_ID, 
                                     'Multi Step': 1, 
                                     'Response': rsp,
                                     'Date Time': d })
        
                    # LLM query and response for the prompt of turn 2
                    rsp = LLMAggregator.process_conversation(self.is_model_open_source(), self.get_llm_index(), self.get_temp(), conversations, row['Present Turn 2'])
                    d = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    conversations.append({"role": "assistant", "content": rsp})
                    # We create a new row in the output file with the prompt 2 request (in present) and the generated response from the LLM along with other useful info.
                    new_rows.append({'Prompt': row['Prompt'], 
                                     'New Prompt': row['Present Turn 2'], 
                                     'SemanticCategory': row["SemanticCategory"], 
                                     'SubTopic': row['SubTopic'],
                                     'QueryType': row['QueryType'],
                                     'Model': self.get_llm(),              
                                     'Version': self.get_llm_version(),
                                     'Temp.': self.get_temp(),
                                     'Unique ID': Row_ID, 
                                     'Multi Step': 2, 
                                     'Response': rsp,
                                     'Date Time': d })
                    # If it's 3 turn attack, we continue to query the LLM and get the response for the prompt of turn 3 in present tense
                    if nb_iterations == 3:
                        rsp = LLMAggregator.process_conversation(self.is_model_open_source(), self.get_llm_index(), self.get_temp(), conversations, row['Present Turn 3'])
                        d = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # We create a new row in the output file with the prompt 3 request (in present) and the generated response from the LLM along with other useful info.
                        conversations.append({"role": "assistant", "content": rsp})
                        new_rows.append({'Prompt': row['Prompt'], 
                                     'New Prompt': row['Present Turn 3'], 
                                     'SemanticCategory': row["SemanticCategory"], 
                                     'SubTopic': row['SubTopic'],
                                     'QueryType': row['QueryType'],
                                     'Model': self.get_llm(),              
                                     'Version': self.get_llm_version(),
                                     'Temp.': self.get_temp(),
                                     'Unique ID': Row_ID, 
                                     'Multi Step': 3, 
                                     'Response': rsp,
                                     'Date Time': d })
                    
                count = count + 1
            
            # Create the new DataFrame with new rows and columns
            output_df = pd.DataFrame(new_rows)
    
            # Make sure the directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
            # Check if the file already exists
            file_exists = os.path.exists(output_file)
    
            # Write the new DataFrame to the output CSV file in a 'append' mode while ignoring the first row (for column names)
            output_df.to_csv(output_file, mode='a', header=not file_exists, index=False)
            print(f"LLM responses written to {output_file}.")
    
        except FileNotFoundError:
            print(f"The file {LLMAggregator.MULTITURN_DATASETS[""+str(nb_iterations)]} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Full traceback:")
            traceback.print_exc()  # This will print the detailed traceback
