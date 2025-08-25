def parse_value(value):
    """
    Attempt to convert string value to corresponding Python literal.
    If the conversion fails the original string is returned.
    """
    import ast
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_possible_list(value):
    """
    Parse input that is either a list or a single int or float.
    """
    import ast
    if not isinstance(value, str):
        return [value]

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except Exception:
        pass

    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        items = inner.split(',')
        parsed_items = []
        for item in items:
            item = item.strip()
            if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                item = item[1:-1].strip()
                parsed_items.append(item)
            else:
                try:
                    parsed_items.append(float(item))
                except ValueError:
                    parsed_items.append(item)
        return parsed_items

    return [value]


def to_camel_case(s):
    """
    Convert a snake_case string to CamelCase. For example, "black_scholes" becomes "BlackScholes".
    """
    return ''.join(word.capitalize() for word in s.split('_'))


def load_class(model_name):
    """
    Dynamically load class based on provided model name.
    """
    import importlib
    module_name = model_name.lower()
    class_name = to_camel_case(model_name)
    module = importlib.import_module(f"models.{module_name}")
    return getattr(module, class_name)


def list_files_excluding(directory, exclude_files=None):
    """
    List files in a directory excluding those in the exclude_files list.
    """
    import os
    if exclude_files is None:
        exclude_files = []
    elif not isinstance(exclude_files, list):
        exclude_files = [exclude_files]
    exclude_files.append('__pycache__')
    return [f for f in os.listdir(directory) if f not in exclude_files]
