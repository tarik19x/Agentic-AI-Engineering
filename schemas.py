from typing import List
from pydantic import BaseModel, Field #Basemodel is used to create a model for the data that we want to store in the database. Field is used to define the fields of the model.
#field is used to define the fields of the model (generate metadata), it takes the name of the field as an argument and returns a Field object that can be used to define the field in the model.