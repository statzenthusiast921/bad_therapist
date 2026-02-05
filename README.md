# Bad Therapist

## Description

The purpose of this project was to:

- have some fun learning about RAG systems
- apply my newly acquired RAG skills in a unique manner
- learn tips/tricks about prompt engineering
- learn about different small and large language models
- learn about vector databases
- utilize AI tools like Cursor to help support building this app


## Data
The [initial dataset](https://github.com/statzenthusiast921/bad_therapist/blob/main/scripts/question_answer_db.py) used in Pinecone was generated from a prompt I designed to create a list of 50 typical questions one would ask a therapist and 50 responses to those questions in character as a narcissistic therapist. 


### App
Click [here](https://bad-therapist-jz.onrender.com/) to view the app.  It might need a few minutes to "wake up" if it has been idle for awhile.

## Challenges
- Developing a solid prompt was the most challenging task of this entire project.  I wanted to encase Dr. Vain's responses in something that had the elements of helpful advice, but when put together came off snooty, dismissive, and self-obsessed. I was unable to find the right balance of too helpful vs. not helpful enough.  Thus, I decided to fully embrace his narcissism and make Dr. Vain generate plainly unhelpful responses.
- Once the app was finished, the deployment process proved to be very difficult and forced me to make some quick, last-minute changes to the underlying infrastructure of Dr. Vain. I originally planned to run the model locally through Ollama, but I quickly hit a wall with hosting. Render’s free tier has a 500MB memory limit, which isn't nearly enough to load a powerful LLM. By switching the processing to Groq (not to be confused with X/Twitter's Grok), I was able to offload all the heavy lifting to their cloud servers, keeping the app lightweight enough to stay within Render's limits while still getting instant, high-end performance.
- This application offered opportunities to use AI IDE tools like Cursor to enhance development efforts. For example, I have no experience using Javascript and was able to utilize the base models available in Cursor to utilize Javascript with the zooming in and out of pictures.   




