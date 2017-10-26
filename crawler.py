
import time
from ghost import Ghost, Session
ghost = Ghost()

url = "http://www.rt.com/usa/406881-trump-says-total-termination-of-iran-deal-possible/"
url2 = "https://economictimes.indiatimes.com/news/politics-and-nation/centre-asks-states-not-to-deny-pds-food-to-poor-sans-aadhaar/articleshow/61241724.cms"

comment_url = "https://spoxy-shard5.spot.im/v2/spot/sp_6phY2k0C/post/%1/"

news_id = "406881"

USERAGENT = "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:45.0) Gecko/20100101 Firefox/45.0"

session = Session(ghost, download_images=False, display=True, user_agent=USERAGENT)

page, extra_resources = session.open(url)
session.show()
time.sleep(5)

with open('content2.html', 'w') as f:
    f.write(session.content)

# gracefully clean off to avoid errors
session.webview.setHtml('')
session.exit()

print("Done")

#commentBox.show(61241724);



