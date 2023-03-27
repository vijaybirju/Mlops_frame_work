import sys

def error_massage_details(error,error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_massage="Error occured in phton script name [{0}] line number [{1}] error massage[{2}]".format(
    file_name,exc_tb.tb_lineno,str(error)
    )