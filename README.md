# findata_service

Example of a RESTful API service to query financial data

## Instructions to Run

1. **Create a Conda Virtual Environment:**

   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the Virtual Environment:**

   - _For Windows:_

     ```bash
     conda activate <your_env_name>
     ```

   - _For macOS/Linux:_
     ```bash
     source activate <your_env_name>
     ```

3. **Run the API Service:**
   ```bash
   uvicorn main:app --reload
   ```

Now, the RESTful API service should be up and running.

Feel free to customize the instructions or provide more details based on your specific requirements.
