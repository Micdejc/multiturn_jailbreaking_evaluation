import os
from llm_aggregator import LLMAggregator

"""   
INFO ABOUT THIS SOURCE CODE

This source code defines functions for a superclass that manages the targeted LLM (model) and the parameters used in experiments.


"""


class MyModel:


    # Function of class initialization (generator)
    # Input:
    # is_selected_llm_open: True if the selected model is open-Source LLM, false if it's a close-source LLM.
    # selected_llm_index: The index of the selected LLM to use from the list, by default the 1st LLM from opensource LLMs is selected
    # selected_temp_index: The index of the selected temperature to use from the list, by default the 1st temperature from the list is selected
    def __init__(self, is_selected_llm_open = True, selected_llm_index = 0, selected_temp_index = 0):
        
        # Depending on the selected LLM (open or close), we select the corresponding list
        if is_selected_llm_open:
            llm_list = LLMAggregator.OPEN_LLM_VERSIONS
        else:
            llm_list = LLMAggregator.CLOSE_LLM_VERSIONS
            #if it's a close-source model chosen, then we also load its corresponding API key
            os.environ["API_KEY"] = LLMAggregator.API_KEYS[selected_llm_index]
            
        # We keep the type of selected model (open or close) and its index in the list    
        self.IS_SELECTED_LLM_OPEN = is_selected_llm_open
        self.SELECTED_LLM_INDEX = selected_llm_index 
        self.SELECTED_MODEL = llm_list[selected_llm_index]    
        self.MODEL_NAME = self.SELECTED_MODEL["MODEL_NAME"]
        self.LLM = self.SELECTED_MODEL["LLM"]
        self.LLM_VERSION = self.SELECTED_MODEL["LLM_VERSION"]

        #SELECTED_TEMPTEMP is the choosen value of temperature for the current experiment which is the first value (0.1) by default from the list
        self.SELECTED_TEMP = LLMAggregator.TEMPS[selected_temp_index]


    # Method to update the LLM model index and model name.
    # Input:
    # is_selected_llm_open: True if the selected model is open-Source LLM, False if it's a close-source LLM.
    # newIndex: The new index of the selected LLM to use from the list, by default the first is selected
    def set_llm_index(self, is_selected_llm_open=True, newIndex: int = 0):
        
        # Depending on the selected LLM (open or close), we select the corresponding list
        if is_selected_llm_open:
            llm_list = LLMAggregator.OPEN_LLM_VERSIONS
        else:
            llm_list = LLMAggregator.CLOSE_LLM_VERSIONS
            #if it's a close-source model chosen, then we also load its corresponding API key
            os.environ["API_KEY"] = LLMAggregator.API_KEYS[newIndex]
            
        if 0 <= newIndex < len(llm_list):
            # We keep the type of selected model (open or close) and its index in the list    
            self.IS_SELECTED_LLM_OPEN = is_selected_llm_open
            self.SELECTED_LLM_INDEX = newIndex 
            self.SELECTED_MODEL = llm_list[newIndex]
            self.MODEL_NAME = self.SELECTED_MODEL["MODEL_NAME"]
            self.LLM = self.SELECTED_MODEL["LLM"]
            self.LLM_VERSION = self.SELECTED_MODEL["LLM_VERSION"]
            print(f"LLM updated to: {self.LLM} {self.LLM_VERSION}")
        else:
            raise ValueError(f"Invalid LLM index: {newIndex}. Must be 0 to {len(llm_list)-1}.")

    # Method to update the temperature index and value.
    # Input:
    # newIndex: The new index of the selected temperature to use from the list
    def set_temperature_index(self, newIndex: int):
        if 0 <= newIndex < len(LLMAggregator.TEMPS):
            self.SELECTED_TEMP = LLMAggregator.TEMPS[newIndex]
            print(f"Temperature updated to: {self.SELECTED_TEMP}")
        else:
            raise ValueError(f"Invalid temperature index: {newIndex}. Must be 0 to {len(LLMAggregator.TEMPS)-1}.")    
    
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

    # Get the selected temperature for the targeted model
    def get_temp(self):
        return self.SELECTED_TEMP

    