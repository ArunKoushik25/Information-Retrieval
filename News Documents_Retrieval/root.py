import os
import json

MAX_TERM_LENGTH = 25  # Set your desired maximum term length

def create_structure(node, term, posting_list):
    if not term:
        node["posting_list"] = posting_list
        return

    first_char, rest_of_term = term[0], term[1:]
    if first_char not in node:
        node[first_char] = {}

    create_structure(node[first_char], rest_of_term, posting_list)

def create_directory_structure_from_json(json_file, root):
    with open(json_file, "r") as f:
        inverted_index = json.load(f)

    for term, posting_list in inverted_index.items():
        if len(term) <= MAX_TERM_LENGTH:
            create_structure(root, term, posting_list)

def save_structure_to_disk(node, current_path=""):
    for char, child_node in node.items():
        new_path = os.path.join(current_path, char)
        if char == "posting_list":
            with open(new_path + ".json", "w") as f:
                json.dump(child_node, f)
        else:
            os.makedirs(new_path, exist_ok=True)
            save_structure_to_disk(child_node, new_path)

# Example usage
root = {}
json_file_path = "inverted_index.json"

create_directory_structure_from_json(json_file_path, root)
save_structure_to_disk(root)