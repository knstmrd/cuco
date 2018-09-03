# CUCO

CUCO (collective universal conscious organism) was a project developed at the [2018 Tallinn Summer School on experimental interaction design (Saint-Petersburg, 26-31 August 2018)](http://summerschool.tlu.ee/russia/).

The project was developed within the framework of the [Neurotheater concept](http://news.ifmo.ru/en/news/7799/), in which various sensors (heartrate, EEG, skin conductance, etc.) are used as part of a theatrical performance.

In this project, the MUSE sensor was used to record EEG data from participants as they looked at 5 paintings. Then, LightGBM was used to build a predictive model which would predict what painting a person is looking at based on their brain activity. This was then used in real-time to affect the lighting on stage -- brain activity of several people was gathered via MUSE headsets, blended in MAX/MSP, sent via OSC to a python script, which would classify one-second chunks and output a color scheme (based on the original paintings used in training). The color scheme was passed via OSC to TouchDesigner, where a real-time video for the backdrop was generated.

The code in this repo includes the feature extraction, training, and real-time prediction and message passing. Due to severe time constraints, and the impossibility of getting data unaffected by external factors (noise, lighting, etc.), this is more of a sketch rather than a real predictive model, however, real-life tests showed that it's predictions are better than random guesses.