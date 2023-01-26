from web_routing import WebRouting

# Defining main function
def main():
    print("Starting Server")
    web_routing = WebRouting().getInstance()
    web_routing.app.run(port = 8081)
  
  
# Using the special variable 
# __name__
if __name__=="__main__":
    main()