import os
import json

TASK_LIST = ['CodeTrans', 'CodeSearchNet', 'BFP', 'CONCODE']
BASE_DIR = "configs/CodeTask"

def create_config():

    os.makedirs(BASE_DIR, exist_ok=True)

    for idx, task in enumerate(TASK_LIST):
        folder_name = task 
        task_path = os.path.join(BASE_DIR, folder_name)
        os.makedirs(task_path, exist_ok=True)

        # train_tasks.json & dev_tasks.json (only include its own dataset)
        task_entry = {
            "CodeTask": [
                {
                    "sampling strategy": "full",
                    "dataset name": task
                }
            ]
        }

        for split in ["train", "test"]:
            with open(os.path.join(task_path, f"{split}_tasks.json"), "w", encoding="utf-8") as f:
                json.dump(task_entry, f, indent=2)

        # test_tasks.json includes all tasks
        dev_entry = {
            "CodeTask": [
            ]
        }

        with open(os.path.join(task_path, "dev_tasks.json"), "w", encoding="utf-8") as f:
            json.dump(dev_entry, f, indent=2)
    print(f"Configuration files created in {BASE_DIR}")
    

if __name__ == "__main__":
    create_config()