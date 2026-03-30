from __future__ import annotations
from dataclasses import dataclass, field #makes class creation easier, no need to write self that much in __init__
from typing import ClassVar, List


class Segment():
    def __init__(self, start_phrase: str = "No startphrase given", end_phrase: str = "No endphrase given", text: str = "", description: str = ""):
        self.start_phrase: str = start_phrase
        self.end_phrase: str = end_phrase
        self.text: str = text
        self.description: str = description
        self.network = None

    def __repr__(self):
        return f"""{self.description} \n {self.start_phrase} \n- {self.end_phrase} \n 
        There is text: {bool(self.text)} \n There is network: {bool(self.network)}\n"""

    def __str__(self):
        return self.description if self.description else f"{self.start_phrase} - {self.end_phrase}"


    @classmethod
    def from_dict(cls, scene: dict):

        sp = scene.get("start_phrase", "")
        ep = scene.get("end_phrase", "")
        desc = scene.get("description", "")
        txt = scene.get("text", "") 
        
        
        return cls(start_phrase=sp, end_phrase=ep, text=txt, description=desc)


        

    
    
