#!/usr/bin/env python3
import requests
import os

class Scrapper() : 
    '''
    Télécharger les fichiers d'archives mensuelles météo de météo France depuis 2010'
    '''
    def __init__(self) : 
        self.base_archiche_url = "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Nivo/Archive/" # nomenclature nom de fichier : "nivo.aaaamm.csv.gz"
        self.storage = "archive_data/"

        if not os.path.exists(self.storage):
            os.makedirs(self.storage)

    def run(self) : 
        year = 2010
        month = 1 
        while year <= 2021 : 
            if year != 2021 : 
                for i in range(12) : 
                    file_name = self.build_file_name(year, i+1)
                    self.get_archive_file(file_name)
            else : 
                for i in range(7) : 
                    file_name = self.build_file_name(year, i+1)
                    self.get_archive_file(file_name)
            year += 1

    def get_archive_file(self, file_name) : 
        url = "{}{}.gz".format(self.base_archiche_url, file_name)
        output_file = "{}{}".format(self.storage, file_name)

        response = requests.get(url)
        if response.status_code == 200 :
            print("Download file {} >>>> {}".format(url, output_file))
            with open(output_file, "wb") as _buffer :
                _buffer.write(response.content)


    def build_file_name(self, year, month) : 
        prefix = "nivo"
        ext = "csv"

        if month < 10 : 
            return "{}.{}0{}.{}".format(prefix,year,month, ext)
        else : 
            return "{}.{}{}.{}".format(prefix,year,month, ext)


if __name__=="__main__":
    s = Scrapper()
    s.run()

