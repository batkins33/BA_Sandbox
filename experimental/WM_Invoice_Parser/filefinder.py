import os

folder = r"C:\Users\brian.atkins\OneDrive - Lindamood Demolition\24-105 PHMS NPC - Documents\PM\Invoices\invoices"

for f in os.listdir(folder):
    print(repr(f))
