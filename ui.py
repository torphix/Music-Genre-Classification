import os
import sys

if __name__ == '__main__':
        # sys.argv = ["python3", "-m", "venv", "venv", "&&", 
        #             "source", "venv/bin/activate", "&&", 
        #             "pip", "install", "-r", "requirements/requirements_ui.txt", "&&",
        #             "streamlit", "run", "src/frontend/frontend.py"]
        os.chdir('src')
        os.system('streamlit run frontend/frontend.py')