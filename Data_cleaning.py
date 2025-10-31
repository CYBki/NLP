# Normalize whitespace
text = "Hello,   World!   2025"

"""
print(text.split())
#output: ['Hello,', 'World!', '2025']
"""

cleaned_text1 = ' '.join(text.split())
print(cleaned_text1)
print(f"text : {text} \n cleaned_text1: {cleaned_text1}")

###########################################################
# Lowercase all text
cleaned_text2 = text.lower()
print(f"text : {text} \n cleaned_text2: {cleaned_text2}")
###########################################################

# Strip punctuation
import string
cleaned_text3 = text.translate(str.maketrans('', '', string.punctuation))
print(f"text : {text} \n cleaned_text3: {cleaned_text3}")
###########################################################
# Strip non-alphanumeric characters, %, @, /, *, #
import re
cleaned_text4 = re.sub(r'[^a-zA-Z0-9\s%/@*#]', '', text)
print(f"text : {text} \n cleaned_text4: {cleaned_text4}")
###########################################################
# Perform spell correction

from textblob import TextBlob
textn = "Hellio, Wirld! 2025"
cleaned_text5 = TextBlob(textn).correct()
print(f"text : {textn} \n cleaned_text5: {cleaned_text5}")
###########################################################
# Strip HTML/URL tags

from bs4 import BeautifulSoup
html_text = text = "<div>Hello,   World!   2025</div>"
cleaned_text6 = BeautifulSoup(html_text, "html.parser").get_text()
print(f"text : {html_text} \n cleaned_text6: {cleaned_text6}")
###########################################################