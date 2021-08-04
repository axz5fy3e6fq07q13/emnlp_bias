import wikipedia
from bs4 import BeautifulSoup
import json
import unidecode
import re
import requests
from wikidata.client import Client
import spacy
import time

nlp = spacy.load('en_core_web_sm')

client = Client()

def generate_question(topic):
	topic = topic.replace("_", " ")
	try:
		context = wikipedia.summary(topic)
	except (wikipedia.exceptions.DisambiguationError,wikipedia.DisambiguationError) as e:
		s = e.options[0]
		context = wikipedia.summary(s)
	except:
		return None
	doc = nlp(context)
	# assert doc.has_annotation("SENT_START")
	first_sent = list(doc.sents)[0]
	for i,t in enumerate(first_sent):
		if t.text in ['is','was','are','were']:
			span = str(first_sent[i+1:])
			question = "Who discovered " + span
			return question
		elif t.text in ['describes','states','applies','establishes','gives']:
			span = str(first_sent[i:])
			question = "Who discovered a theorem that " + span
			return question
	if "used to" in str(first_sent):
		return "What equation can be used to {}".format(first_sent.split("used to")[1])

	return None

def get_wiki_summary(s):
	try:
		p = wikipedia.summary(s)
		return p
	except (wikipedia.exceptions.DisambiguationError,wikipedia.DisambiguationError) as e:
		s = e.options[0]
		p = wikipedia.summary(s)
		return p
	except:
		return s.replace(" ","_")

def get_wiki_name(wiki_guess):
	try:
		p = wikipedia.summary(wiki_guess)
		return wiki_guess
	except wikipedia.DisambiguationError as e:
		return e.options[0]
	except:
		return ''

def get_gender_nationality(wiki_name):
	try:
		a = requests.get("https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles={}&normalize=1&format=json".format(wiki_name))
		a = a.json()
		wiki_id = list(a['entities'].keys())[0]
		return get_wikidata(wiki_id)
	except:
		return {'gender': 'unknown', 'ethnicity': 'unknown','nationality': 'unknown'}

def get_wikidata(wiki_id):
	entity = client.get(wiki_id, load=True)
	p = client.get('P21')
	gender = "unknown"
	if p in entity:
		gender = str(entity[p].label)
	p = client.get('P172')
	ethnicity = "unknown"
	if p in entity:
		ethnicity = str(entity[p].label)
	p = client.get('P27')
	nationality = 'unknown'
	if p in entity:
		nationality = str(entity[p].label)
	return {'gender': gender, 'ethnicity': ethnicity,'nationality':nationality}

def find_names(law_name):
	law_name = law_name.replace("_", " ")
	names = law_name.split("-")
	name1 = names[0].split(" ")[-1].strip("() ,;")
	name2 = names[1].split(" ")[0].strip("() ,;")
	links = wikipedia.page(law_name).links
	summary = str(wikipedia.summary(law_name)).lower()
	potential_names = [i for i in links if ("mechanics" not in i.lower() and (name1.lower() in i.lower() or name2.lower() in i.lower()) and "–" not in i.lower())]
	return [i for i in potential_names if i.lower() in summary]


data = wikipedia.page("List_of_scientific_laws_named_after_people").html()
soup = BeautifulSoup(data,'lxml')

laws = []
people = []

start = time.time()
print("Generating laws")

exceptions = ["Cauchy's integral formula"]

for items in soup.find('table', class_='wikitable').find_all('tr')[1::1]:
	data = items.find_all(['th','td'])
	if " and" in data[2].text and data[0].a.text not in exceptions:
		laws.append(data[0].a.text)
		people.append(data[2].text.strip())

data = wikipedia.page("Scientific_phenomena_named_after_people").content.split("\n")
for line in data:
	line = line.replace(chr(8211),chr(45))
	temp_line = line.split(" - ")
	if len(temp_line)>1 and " and" in temp_line[1]:	
		laws.append(temp_line[0])
		people.append(temp_line[1])

for i in range(len(laws)):
	if "(a.k.a" in laws[i]:
		laws[i] = laws[i].split("(a.k.a")[0]

quizbowl = open("quizbowl_eponymous.txt").read().strip().split("\n")
quizbowl_people = [find_names(i) for i in quizbowl]

quizbowl = [quizbowl[i] for i in range(len(quizbowl)) if len(quizbowl_people[i])==2]
quizbowl_people = ["{} and {}".format(quizbowl_people[i][0],quizbowl_people[i][1]) for i in range(len(quizbowl_people)) if len(quizbowl_people[i]) == 2]

laws+=quizbowl
people+=quizbowl_people

print(people[-1],laws[-1])

print("Took {} time to generate laws".format(time.time()-start))
start = time.time()
print("Generating questions")
questions = [generate_question(i) for i in laws]

print("We have {} questions".format(len(questions)))
print("Took {} time to generate questions".format(time.time()-start))

start = time.time()
print("Finding demographic info on people")
for i in range(len(people)):
	if "(" in people[i]:
		people[i] = people[i].split("(")[0].strip()
	people[i] = people[i].split(",")
	last_people = people[i][-1].split(" and")
	people[i] = people[i][:-1] + last_people
	for j in range(len(people[i])):
		people[i][j] = people[i][j].strip() 
	people[i] = [j for j in people[i] if j!='']

	for j in range(len(people[i])):
		gender_info = get_gender_nationality(get_wiki_name(people[i][j].replace(" ","_")))
		gender_info['name'] = people[i][j]
		for k in gender_info:
			gender_info[k] = unidecode.unidecode(gender_info[k])
		people[i][j] = gender_info
print("Took {} time to find demographic info on people".format(time.time()-start))
start = time.time()
print("Finding context for questions")
contexts = []
for law_name in laws:
	summary = get_wiki_summary(law_name)
	summary = re.sub(r'\n'," ",summary)
	summary = re.sub(r' +', ' ',summary)
	summary = re.sub(r'\S+\\\S+', '',summary)
	summary = re.sub(r' +', ' ',summary)
	summary = summary.split(" ")
	better_summary = []
	for i in range(len(summary)):
		if len(summary[i]) == 1:
			if summary[i] != 'a':
				continue
			elif i<len(summary)-1 and len(summary[i+1]) == 1:
				continue
			elif i>0 and len(summary[i-1]) == 1:
				continue
			else:
				better_summary.append(summary[i])
		elif '{' in summary[i] or '}' in summary[i] or '\\\\' in summary[i]:
			continue
		else:
			better_summary.append(summary[i])
	contexts.append(" ".join(better_summary))
print("Took {} time to find context".format(time.time()-start))

print("Writing to file")
with open("doubly_eponymous_hard.jsonl","w") as w:
	for i in range(len(laws)):
		if contexts[i]!='' and questions[i]!=None:
			d = {'name': unidecode.unidecode(laws[i]), 'question': unidecode.unidecode(questions[i]),
				'context': unidecode.unidecode(contexts[i]).strip(), 'options': [j for j in people[i]]}
		w.write(json.dumps(d))
		w.write('\n')

