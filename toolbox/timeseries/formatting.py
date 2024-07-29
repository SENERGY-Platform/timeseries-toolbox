import re

DATETIME_FORMATS = [
    ("yyyy-MM-ddTHH:mm:ss.SSSZ", "^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"), #2023-06-01T12:34:56.789Z
    ("yyyy-MM-ddTHH:mm:ssZ", "^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"), #2023-06-01T12:34:56Z
    ("yyyy-MM-ddTHH:mm:ss", "^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"), #2023-06-01T12:34:56
    ("yyyy-MM-dd", "^\d{4}-\d{2}-\d{2}$"), #2023-06-01
    ("yyyyMMddTHHmmssZ", "^\d{8}T\d{6}Z$"), #20230601T123456Z
    ("MM/dd/yyyy", "^\d{2}/\d{2}/\d{4}$"), #06/01/2023
    ("dd/MM/yyyy", "^\d{2}/\d{2}/\d{4}$"), #01/06/2023
    ("dd-MM-yyyy", "^\d{2}-\d{2}-\d{4}$"), #01-06-2023
    ("yyyy-MM-dd HH:mm:ss", "^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"), #2023-06-01 12:34:56
    ("MM/dd/yyyy HH:mm:ss", "^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$"), #06/01/2023 12:34:56
    # ("unix_seconds", "^\d{10}$"), #1622543996
    # ("unix_milliseconds", "^\d{13}$"), #1622543996000
    ("Day, dd Mon yyyy HH:mm:ss Z", "^\w{3}, \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2} [+-]\d{4}$") #Wed, 01 Jun 2023 12:34:56 +0000
]

def get_ts_format_from_str(ts):
    for format in DATETIME_FORMATS:
        if bool(re.search(format[1], ts)):
            return format[0]
    return "unix"