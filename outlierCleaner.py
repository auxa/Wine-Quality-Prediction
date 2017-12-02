def outlierCleaner(dataPoints):

    #calculate the error,make it descend sort, and fetch 90% of the data
    
    errors = (net_worths-predictions)**2
    cleaned_data =zip(ages,net_worths,errors)
    cleaned_data = sorted(cleaned_data,key=lambda x:x[2][0], reverse=True)
    limit = int(len(net_worths)*0.1)

    return cleaned_data[limit:]
