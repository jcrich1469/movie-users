import json
import os
from filelock import Timeout, FileLock

class FileManager:

    @staticmethod
    def write_file(file_path):
        """Creates a new empty file."""
        try:
            with open(file_path, 'w'):
                pass
            print(f"File '{file_path}' created successfully")
        except IOError as e:
            print(f"An error occurred while creating the file: {e}")

    @staticmethod
    def remove_file(file_path):
        """Removes a file."""
        try:
            os.remove(file_path)
            print(f"File '{file_path}' removed successfully")
        except FileNotFoundError:
            print(f"File '{file_path}' not found")
        except PermissionError as e:
            print(f"Permission denied: {e}")
        except OSError as e:
            print(f"An error occurred while removing the file: {e}")

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

    @staticmethod
    def write_queue_to_file(queue, file_path):
        """Writes the contents of a queue to a text file."""
        try:
            with open(file_path, 'w') as file:
                for item in queue.items:
                    file.write(str(item) + '\n')
            print(f"Queue successfully written to {file_path}")
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")

    @staticmethod
    def read_queue_from_file(file_path):
        """Reads the contents of a text file and initializes a queue."""
        try:
            with open(file_path, 'r') as file:
                queue = FileQueue()
                for line in file:
                    item = line.strip()
                    queue.enqueue(item)
                print(f"Queue successfully loaded from {file_path}")
                return queue
        except IOError as e:
            print(f"An error occurred while reading from the file: {e}")

    @staticmethod
    def enqueue_to_file(item, file_path):
        """Enqueues an item to the end of a text file."""
        try:
            with open(file_path, 'a') as file:
                file.write(str(item) + '\n')
            print(f"Item '{item}' successfully enqueued to {file_path}")
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")
    #todo abstract better?
    todo_lock = FileLock("todo.txt.lock")

    @staticmethod
    def dequeue_from_file(file_path):
        """Dequeues an item from the beginning of a text file in a thread-safe manner."""
        FileManager.todo_lock.acquire()
        try:
            with open(file_path, 'r+') as file:
                lines = file.readlines()
                if lines:
                    item = lines[0].strip()
                    file.seek(0)
                    file.writelines(lines[1:])
                    file.truncate()
                    print(f"Item '{item}' successfully dequeued from {file_path}")
                else:
                    item = None
                    print(f"File '{file_path}' is empty, unable to dequeue")
        except IOError as e:
            print(f"An error occurred while reading from or writing to the file: {e}")
            item = None
        finally:
            FileManager.todo_lock.release()
        return item


    @staticmethod
    def write_stack_to_file(stack, file_path):
        """Writes the contents of a stack to a text file."""
        try:
            with open(file_path, 'w') as file:
                for item in stack.items[::-1]:  # Write items in reverse order
                    file.write(str(item) + '\n')
            print(f"Stack successfully written to {file_path}")
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")

    @staticmethod
    def read_stack_from_file(file_path):
        """Reads the contents of a text file and initializes a stack."""
        try:
            with open(file_path, 'r') as file:
                stack = StackFile()
                for line in file:
                    item = line.strip()
                    stack.push(item)
                print(f"Stack successfully loaded from {file_path}")
                return stack
        except IOError as e:
            print(f"An error occurred while reading from the file: {e}")
        
    @staticmethod
    def push_to_file(item,file_path):
        """Pushes an item to the stack file."""
        try:
            with open(file_path, 'a') as file:
                file.write(str(item) + '\n')
            print(f"Item '{item}' pushed to stack file '{file_path}'")
        except IOError as e:
            print(f"An error occurred while pushing item to stack file: {e}")

    @staticmethod
    def pop_from_file(file_path):
        """Pops an item from the stack file."""
        try:
            with open(file_path, 'r+') as file:
                lines = file.readlines()
                if lines:
                    item = lines.pop()
                    file.seek(0)
                    file.truncate()
                    file.writelines(lines)
                    print(f"Item '{item.strip()}' popped from stack file '{file_path}'")
                    return item.strip()
                else:
                    print(f"Stack file '{file_path}' is empty")
                    return None
        except IOError as e:
            print(f"An error occurred while popping item from stack file: {e}")
            return None

    @staticmethod
    def lines_in_file(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return len(lines)

    @staticmethod
    def is_empty(file_path):
        try:
            # Open the file in read mode
            with open(file_path, 'r') as file:
                # Attempt to read the first line
                first_line = file.readline()
                # If the first line does not exist, the file is empty
                return not first_line
        except FileNotFoundError:
            print(f"The file {file_path} does not exist.")
            return True  # Considering non-existent files as empty
        except Exception as e:
            print(f"An error occurred: {e}")
            return True  # Error scenario, handling gracefully
    
    

