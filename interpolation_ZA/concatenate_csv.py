import os
import re
import sys
import glob
import csv
import pprint
from pandas import read_csv, DataFrame


class ConcatenateCSV:

    def __init__(self,list_files): 
        self.list_files = list_files

    def Concatenate(self):
        self.dict_tot = {} 
        self.counter = 0

        for f in self.list_files:
            name = os.path.basename(f)

            print('File : %s'%(name))

            # Use pandas to get the dict inside the csv file #
            panda_data = read_csv(f)
            dict_from_file = panda_data.to_dict(orient='list')

            # Initialize dict at first elements #
            if self.counter == 0:
                for key in dict_from_file.keys():
                    self.dict_tot[key] = []
            
            # Append full dict #
            for key,val in dict_from_file.items():
                try:
                    self.dict_tot[key] += val
                except KeyError:
                    sys.exit('Your csv files do not have the same keys')
                entries = len(val)

            self.counter += entries

            print('\tNumber of hyperparameter sets in this file : %d'%(entries)) 

        print('Total number of hyperparameter sets : %d'%(self.counter)) 
        #for line in pprint.pformat(self.dict_tot).split('\n'):
        #    print(line)

        return self

    def WriteToFile(self,name):
        # Write to file #
        self.path_out = os.path.join(os.getcwd(),name)
        invalid_counter = 0
        with open(self.path_out, 'w') as the_file:
            w = csv.writer(the_file)
            # Write keys as header #
            w.writerow(self.dict_tot.keys())
            # Write each test line by line #
            for i in range(0,self.counter): # loop over the elements values of the dict
                test_line = []
                valid_loss = True 
                valid_error = True # If valid, can append to file 
                for key,val in self.dict_tot.items(): # only select the i-th element
                    # check if negative values -> likely an overflow #
                    if key == 'val_loss' and val[i]<0:
                        valid_loss = False
                        print('Val_loss negative (%0.5f), will remove it from full csv file'%(val[i]))
                    # check overflow in eval_error #
                    if key == 'eval_f1score_mean' and val[i]>1000:
                        valid_error = False
                        print('Eval_error too large (%0.2f), will remove it from full csv file'%(val[i]))
                            
                    # Append the number in the line #
                    else:
                        test_line.append(val[i])

                if valid_error:
                    w.writerow(test_line)
                    
                if not valid_error or not valid_loss:
                    invalid_counter += 1
                    print("Negative val_loss or overflow in eval_error")
                    # We don't want these values anyway because eval_error will be the largest 

        print('CSV file saved as %s'%(self.path_out))
        print('Invalid cases (overflow), total %d/%d'%(invalid_counter,self.counter))

def main():
    inputs = sys.argv[1:-1]
    output = sys.argv[-1]
    if os.path.exists(output):
        test = input('Careful, this file already exists, do you want to erase it ? [yes]')
        if test != 'yes':
            sys.exit('Process closed')
    print ('Files you want to concatenate :')
    for name in inputs:
        print ('\t'+name)
    print ('Output of the concatenation :')
    print ('\t'+output)
    instance = ConcatenateCSV(inputs)
    instance.Concatenate()
    instance.WriteToFile(output)

    



if __name__ == '__main__':
    main()



