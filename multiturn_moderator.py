import os
import requests
import pandas as pd
import datetime
import os
import traceback
import re
import importlib
import llm_moderator
import llm_aggregator
importlib.reload(llm_aggregator)
importlib.reload(llm_moderator)
from llm_aggregator import LLMAggregator
from llm_moderator import LLMModerator


"""   

Copyright (c) 2025, Michael Tchuindjang
All rights reserved.

This code was developed as part of a PhD research project in Cybersecurity and Artificial Intelligence, 
supported by a studentship at the University of the West of England (UWE Bristol).

Use of this software is permitted for academic, educational, and research purposes.  
For any commercial use or redistribution, please contact the author for permission.

Disclaimer:
In no event shall the author or UWE be liable for any claim, damages, or other liability arising from the use of this code.

Acknowledgment of the author and the research context is appreciated in any derivative work or publication.



INFO ABOUT THIS SOURCE CODE

This source code provides functions to moderate the output of a Large Language Models (LLM) using another LLM as moderator.
Below are the main functions included:

1. moderate_llm_responses(): Moderates LLM responses using specific rows from a CSV file of the targeted model.
2. calculate_success_rate(): Calculates the jailbreaking success rate based on the judgments of an LLM moderator for a given CSV file.
3. calculate_success_rate_per_cybersubtopic(): calculates the jailbreaking success rate per cyber subtopic based on the judgments of a moderator (Human or LLM) for a given CSV file.

"""


class MultiTurnModerator():

    # Function of class initialization (generator)
    # Input:
    # is_selected_llm_open: True if the selected model is open-Source LLM, false if it's a close-source LLM.
    # selected_llm_index: The index of the selected LLM to use from the list, by default the 1st LLM from opensource LLMs is selected
    def __init__(self, is_selected_llm_open=True, selected_llm_index=0):
        
        # Depending on the selected LLM (open or close) for moderation, we select the corresponding list
        if is_selected_llm_open:
            llm_list = LLMModerator.OPEN_LLM_VERSIONS
        else:
            llm_list = LLMModerator.CLOSE_LLM_VERSIONS
            #if it's a close-source model chosen, then we also load its corresponding API key
            os.environ["API_KEY"] = LLMModerator.API_KEYS[selected_llm_index]
            
        # We keep the type of selected model (open or close) and its index in the list    
        self.IS_SELECTED_LLM_OPEN = is_selected_llm_open
        self.SELECTED_LLM_INDEX = selected_llm_index 
        self.SELECTED_MODEL = llm_list[selected_llm_index]    
        self.MODEL_NAME = self.SELECTED_MODEL["MODEL_NAME"]
        self.LLM = self.SELECTED_MODEL["LLM"]
        self.LLM_VERSION = self.SELECTED_MODEL["LLM_VERSION"]


    # Method to update the LLM model index and model name.
    # Input:
    # is_selected_llm_open: True if the selected model is open-Source LLM, False if it's a close-source LLM.
    # newIndex: The new index of the selected LLM to use from the list, by default the first is selected
    def set_llm_index(self, is_selected_llm_open=True, newIndex: int = 0):
        
        # Depending on the selected LLM (open or close), we select the corresponding list
        if is_selected_llm_open:
            llm_list = LLMModerator.OPEN_LLM_VERSIONS
        else:
            llm_list = LLMModerator.CLOSE_LLM_VERSIONS
            #if it's a close-source model chosen, then we also load its corresponding API key
            os.environ["API_KEY"] = LLMModerator.API_KEYS[newIndex]
            
        if 0 <= newIndex < len(llm_list):
            # We keep the type of selected model (open or close) and its index in the list    
            self.IS_SELECTED_LLM_OPEN = is_selected_llm_open
            self.SELECTED_LLM_INDEX = newIndex 
            self.SELECTED_MODEL = llm_list[newIndex]
            self.MODEL_NAME = self.SELECTED_MODEL["MODEL_NAME"]
            self.LLM = self.SELECTED_MODEL["LLM"]
            self.LLM_VERSION = self.SELECTED_MODEL["LLM_VERSION"]
            print(f"LLM for moderation updated to: {self.LLM} {self.LLM_VERSION}")
        else:
            raise ValueError(f"Invalid LLM index: {newIndex}. Must be 0 to {len(llm_list)-1}.")


    # Getters
    # Get the name of the targeted model for API call
    def get_model_name(self):
        return self.MODEL_NAME

    # Get the short name of the targeted model
    def get_llm(self):
        return self.LLM
        
    # Get the version of the targeted model
    def get_llm_version(self):
        return self.LLM_VERSION
        
    # Check if the targeted model is open or close model
    def is_model_open_source(self):
        return self.IS_SELECTED_LLM_OPEN

    # Get the index of the targeted model from the LLMs' list
    def get_llm_index(self):
        return self.SELECTED_LLM_INDEX




    # Function to moderate LLM responses using specific rows from a CSV file of the targeted model
    # min_row and max_row determine the range of rows to select in the table of input file for optimization of the attack process
    # The name of the column of the moderator (in the csv file) follows the syntax: '<MODEL_NAME> Judge' except for Human moderator which is 'Human'
    # Input:
    # is_targeted_llm_open: Indicates whether the targeted LLM is open or close. It's open-source by default
    # targeted_llm_index: index of the targeted LLM from the List of targets. It's 0 by default
    # nb_iterations: number of iterations to consider. It's 1 (single turn moderation) by default
    # is_past: Indicates whether past or present tense reformulation is considered. It's False (present tense reformulation) by default
    # min_row: Starting index of the rows to select from the input file
    # max_row: Ending index of the rows to select from the input file
    # Output:
    # This function does not return any value but will create a new file, named according to the output_file variable, which will contain the LLM judge moderation for LLM responses
    def moderate_llm_responses(self, is_targeted_llm_open=True, targeted_llm_index=0, nb_iterations=1, is_past=False, min_row=None, max_row=None):

        # We get the relative path of the targeted models depending of the parameters   
        csv_file = self.get_input_file(is_targeted_llm_open, targeted_llm_index, nb_iterations, is_past)
        
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(csv_file)
        
            # Check if 'Response' column exists
            if 'Response' not in df.columns:
                print("The 'Response' column is missing in the input file.")
                return
            # Set defaults if min_row or max_row is None
            if min_row is None:
                min_row = 1
            if max_row is None:
                max_row = len(df)
            if min_row >= max_row:
                print(f"Invalid range: min_row ({min_row}) must be less than max_row ({max_row}).")
                return

            # Except the column of rule-based judgement the column for LLM judgement is name as follows: <MODEL_NAME> Judge 
            if self.get_model_name() == 'Success':
                column_name = self.get_model_name()
            else:
                column_name = self.get_model_name() + ' Judge'
    
            # Check if the column exists, if not, add it
            if column_name not in df.columns:
                df[column_name + ''] = None  # Initialize with None or another default value
        
            # Iterate over the entire DataFrame but only update rows within the range [min_row-1, max_row]
            for index, row in df.iterrows():
                if min_row - 1 <= index <= max_row - 1:
                    # We put judgement score only to output for single turn and the last output of multi-turn
                    
                    if df.loc[index, 'QueryType'] == LLMModerator.REQUEST_TYPE[0]:# If it's instruction-based queries
                        if self.is_model_open_source() and self.get_llm_index() == 0: # If Llama Guard as moderator
                            safety_guideline = LLMModerator.LLAMA_GUARD3_INSTRUCTION_SAFETY_GUIDELINES
                        else:# Since we got only 2 LLMs that we moderate from but this code is to change later...
                            safety_guideline = LLMModerator.GPT4_INSTRUCTION_SAFETY_GUIDELINES
                    else:# If it's script-based queries
                        if self.is_model_open_source() and self.get_llm_index() == 0: # If Llama Guard as moderator
                            safety_guideline = LLMModerator.LLAMA_GUARD3_SCRIPT_SAFETY_GUIDELINES
                        else:# Since we got only 2 LLMs that we moderate from but this code is to change later...
                            safety_guideline = LLMModerator.GPT4_SCRIPT_SAFETY_GUIDELINES                       
                        
                    #For 2 turn, we gather all the conversation from all the steps before the moderation
                    if nb_iterations == 2 and df.loc[index, 'Multi Step'] == nb_iterations:
                        conversation=[]
                        conversation.append({"role": "system", "content": safety_guideline})
                        conversation.append({"role": "user", "content": df.loc[(index-1), 'New Prompt']})
                        conversation.append({"role": "assistant", "content": df.loc[(index-1), 'Response']})
                        conversation.append({"role": "user", "content": df.loc[index, 'New Prompt']})
                        conversation.append({"role": "assistant", "content": df.loc[index, 'Response']})
                        # Pass the conversation to the moderation function
                        df.loc[index, column_name + ''] = LLMModerator.process_moderation(self.is_model_open_source(), self.get_llm_index(), conversation)
                    #For 3 turn, we gather all the conversation from all the steps before the moderation    
                    elif nb_iterations == 3 and df.loc[index, 'Multi Step'] == nb_iterations: 
                        conversation=[]
                        conversation.append({"role": "system", "content": safety_guideline})
                        conversation.append({"role": "user", "content": df.loc[(index-2), 'New Prompt']})
                        conversation.append({"role": "assistant", "content": df.loc[(index-2), 'Response']})
                        conversation.append({"role": "user", "content": df.loc[(index-1), 'New Prompt']})
                        conversation.append({"role": "assistant", "content": df.loc[(index-1), 'Response']})
                        conversation.append({"role": "user", "content": df.loc[index, 'New Prompt']})
                        conversation.append({"role": "assistant", "content": df.loc[index, 'Response']})
                        # Pass the conversation to the moderation function
                        df.loc[index, column_name + ''] = LLMModerator.process_moderation(self.is_model_open_source(), self.get_llm_index(), conversation)
    
                    elif nb_iterations == 1 and df.loc[index, 'Multi Step'] == 0:
                        # For single turn, directly moderate the question/response conversation
                        conversation=[]
                        conversation.append({"role": "system", "content": safety_guideline})
                        conversation.append({"role": "user", "content": df.loc[index, 'Prompt']})
                        conversation.append({"role": "assistant", "content": df.loc[index, 'Response']})
                        df.loc[index, column_name + ''] = LLMModerator.process_moderation(self.is_model_open_source(), self.get_llm_index(), conversation)
        
            # Save the updated dataframe back to the CSV file
            df.to_csv(csv_file, index=False)
            print(f"{self.get_model_name()} scores written to {csv_file}.")
            
            # We automatically calculate and print the success rate straight away after the moderation
            self.calculate_success_rate(is_targeted_llm_open, targeted_llm_index, nb_iterations, is_past)
            
        except FileNotFoundError:
            print(f"The file {csv_file} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Full traceback:")
            traceback.print_exc()  # This will print the detailed traceback




    #Function that calculates the success rate of LLM responses from a CSV file of the targeted model
    # Input:
    # is_targeted_llm_open: Indicates whether the targeted LLM is open or close. It's open-source by default
    # targeted_llm_index: index of the targeted LLM from the List of targets. It's 0 by default
    # nb_iterations: number of iterations to consider. It's 1 (single turn moderation) by default
    # is_past: Indicates whether past or present tense reformulation is considered. It's False (present tense reformulation) by default
    # Output:
    # This function return informations related to the success rates of jailbreaking
    def calculate_success_rate(self, is_targeted_llm_open=True, targeted_llm_index=0, nb_iterations=1, is_past=False):

        # We get the relative path of the targeted models depending of the parameters   
        csv_file = self.get_input_file(is_targeted_llm_open, targeted_llm_index, nb_iterations, is_past)
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
    
            # Except the column of rule-based judgement the column for LLM judgement is name as follows: <MODEL_NAME> Judge 
            if self.get_model_name() == 'Success':
                column_name = self.get_model_name()
            else:
                column_name = self.get_model_name() + ' Judge'
                
            # Count the number of 1s in the 'LLM Judge' column
            num_ones = (df[column_name + ''] == 1).sum()
            
            # Filter rows where judge is 0 or 1
            judge_mask = df[column_name + ''].isin([0, 1])
    
            # Calculate the total number of judged rows based on 'Multi Step' and valid judge values
            if nb_iterations == 1:
                total_judged_rows = df[judge_mask & (df['Multi Step'] == 0)].shape[0]
            elif nb_iterations == 2:
                total_judged_rows = df[judge_mask & (df['Multi Step'] == 2)].shape[0]
            else:
                total_judged_rows = df[judge_mask & (df['Multi Step'] == 3)].shape[0]
            
            # Calculate the success rate
            if total_judged_rows > 0:
                success_rate = num_ones / total_judged_rows
                print(f"Number of judged rows: {total_judged_rows}")
                print(f"Number of successful attempts: {num_ones}")
                print(f"Success Rate: {success_rate * 100:.2f}%")
            else:
                print("No rows to calculate success rate.")
                
        except FileNotFoundError:
            print(f"The file {csv_file} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Full traceback:")
            traceback.print_exc()  # This will print the detailed traceback





    # Function that calculates the jailbreaking success rate per cyber subtopic based on the judgments of a moderator (Human or LLM) from a CSV file.
    # Input:
    # moderator_index: Index of the Column name of the LLM moderator. It's 0 (Human moderation) by default
    # is_targeted_llm_open: Indicates whether the targeted LLM is open or close. It's open-source by default
    # targeted_llm_index: Index of the targeted LLM from the List of targets. It's 0 by default
    # nb_iterations: number of iterations to consider. It's 1 (single turn moderation) by default
    # is_past: Indicates whether past or present tense reformulation is considered. It's False (present tense reformulation) by default
    # Output:
    # This function return informations related to the success rates per cyber subtopic of jailbreaking
    def calculate_success_rate_per_cybersubtopic(self, moderator_index=0, is_targeted_llm_open=True, targeted_llm_index=0, nb_iterations=1, is_past=False):

        # We get the relative path of the targeted models depending of the parameters   
        csv_file = self.get_input_file(is_targeted_llm_open, targeted_llm_index, nb_iterations, is_past)
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Check if the necessary columns exist ie. 'SubTopic' and MODERATOR name column
            if 'SubTopic' not in df.columns or ''+ LLMModerator.MODERATOR_COLUMN_NAMES[moderator_index] not in df.columns:
                print("The required columns 'SubTopic' or '"+ LLMModerator.MODERATOR_COLUMN_NAMES[moderator_index] +"' are missing.")
                return
    
            # Normalize the 'subTopic' column to lowercase for case-insensitive comparison
            df['SubTopic'] = df['SubTopic'].str.lower()
    
            # Fill missing values in 'subTopic' with 'hacking'
            df['SubTopic'] = df['SubTopic'].fillna(LLMModerator.CYBER_SUBTOPICS[0])
    
            # Filter rows with relevant subTopics (case-insensitive)
            filtered_df = df[df['SubTopic'].isin(LLMModerator.CYBER_SUBTOPICS)]
    
            # Filter rows where MODERATOR name column has values 0 or 1
            filtered_df = filtered_df[filtered_df[''+ LLMModerator.MODERATOR_COLUMN_NAMES[moderator_index]].isin([0, 1])]
    
            # Calculate total counts for each subTopic
            total_counts = filtered_df.groupby('SubTopic')[''+ LLMModerator.MODERATOR_COLUMN_NAMES[moderator_index]].count()
    
            # Calculate the number of 1s in MODERATOR name column for each subTopic
            success_counts = filtered_df.groupby('SubTopic')[''+ LLMModerator.MODERATOR_COLUMN_NAMES[moderator_index]].apply(lambda x: (x == 1).sum())
    
            # Calculate the percentage of successful attempts for each subTopic
            percentages = (success_counts / total_counts) * 100

            print(f"Statistics per cybersecurity subtopic for {LLMModerator.MODERATOR_COLUMN_NAMES[moderator_index]} moderation in {csv_file}:\n")
                            
            # Print results: both number and percentage for each subTopic
            for sub_topic in total_counts.index:
                print(f"{sub_topic.capitalize()}:")
                print(f"  Total: {total_counts[sub_topic]}")
                print(f"  Successful attempts: {success_counts[sub_topic]} ({percentages[sub_topic]:.2f}% success rate)")
    
        except FileNotFoundError:
            print(f"The file {csv_file} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Full traceback:")
            traceback.print_exc()  # This will print the detailed traceback



    # Function that retreive the input CSV file from the selected parameters.
    # Input:
    # is_targeted_llm_open: Indicates whether the targeted LLM is open or close. It's open-source by default
    # targeted_llm_index: Index of the targeted LLM from the List of targets. It's 0 by default
    # nb_iterations: number of iterations to consider. It's 1 (single turn moderation) by default
    # is_past: Indicates whether past or present tense reformulation is considered. It's False (present tense reformulation) by default
    # Output:
    # This function return the relative path of the input file of the selected targeted model
    def get_input_file(self, is_targeted_llm_open=True, targeted_llm_index=0, nb_iterations=1, is_past=False):
        
        # Depending on the parameter of the targeted LLM We look for the csv file to moderate
        if is_past:
            suffix = '_past.csv'
        else:
            suffix = '_present.csv'

        # Depending on the targeted LLM (open or close), we select the corresponding list
        if is_targeted_llm_open:
            llm_list = LLMAggregator.OPEN_LLM_VERSIONS
        else:
            llm_list = LLMAggregator.CLOSE_LLM_VERSIONS

        target_model = llm_list[targeted_llm_index]    
        target_model_name = target_model["MODEL_NAME"]
            
        return LLMAggregator.TEST_FOLDER + '' + target_model_name + '/' + target_model_name +'_attack_N_'+ str(nb_iterations) + suffix

