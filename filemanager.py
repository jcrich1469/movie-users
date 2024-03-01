import json

class FileManager:

    @staticmethod
    def read_text_file(file_path):
        """Reads and returns the content of a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except IOError as e:
            print(f"An error occurred while reading the file: {e}")
            return None

    def write_text_file(self, file_path, text):
        with open(file_path, 'w') as file:
            file.write(text)
    
    @staticmethod
    def write_dict_to_file(dictionary, file_path):
        """Writes a dictionary to a file in JSON format."""
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(dictionary,file, ensure_ascii=False, indent=4)
            print(f"Dictionary successfully written to {file_path}")
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")

    @staticmethod
    def load_dict_from_file(file_path):
        """Reads a dictionary from a file which is in JSON format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                dictionary = json.load(file)
                print(f"Dictionary successfully loaded from {file_path}")
                return dictionary
        except IOError as e:
            print(f"An error occurred while reading from the file: {e}")
            return None
