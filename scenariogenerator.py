import pickle
import math
from auction_algorithm import *
from auction_algorithm import buyerinfo, productinfo, auction_instance, price_raising_algorithm
import numpy as np
import random
import itertools
import time
from pprint import pprint
import csv
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import copy

def ___init___(self):
    pass

def generate_scenario(QTY_Buyers,QTY_Items, demandsatisfaction, demand_variance,range_of_valuations_param,supply_variance):
    item_list = []
    buyer_list = []
    i=1
   
    totalsupply= QTY_Items*QTY_Buyers #total supply of all item copies in the market that is distributed among all items (amount of buyers x amount of items)
    

 #############SUPPLY VECTOR####################
    supply=[] #supply vector
    if supply_variance=="S-AVG": #all items have the same supply, which is the average supply
        supply=[round(totalsupply/QTY_Items) for x in range(QTY_Items)]
        
    elif supply_variance=="S-DIST":
        #the supplies are normally distributed around the average supply
        mean=totalsupply/QTY_Items
        std=math.sqrt(totalsupply/QTY_Items)
        
        for x in range(QTY_Items):
            supply.append(round(np.random.normal(mean,std)))
        #if the sum of the supplies is more than 10% higher than the total supply, then the supplies are scaled down
        if sum(supply)>totalsupply*1.1:
            scale_factor=totalsupply/sum(supply)
            supply=[round(supply*scale_factor) for supply in supply]


    else: #supply_variance=="S-RAND":
        volumes = random.choices(range(totalsupply + 1), k=QTY_Items)  # randomly choose n values from 0 t o total_volume
        scale_factor = totalsupply / sum(volumes)  # calculate the scaling factor to ensure the sum of the volumes equals the total_volume
        supply = [max(1,round(volume * scale_factor)) for volume in volumes]  # scale the volumes to ensure the sum equals roughly the total_volume
       
    for x in range(QTY_Items):
        item_list.append(productinfo("P"+str(x+1),supply[x]))
    
    ##########Range of valuations############
    
    valuations_range=range_of_valuations_param*QTY_Items #range of valuations for all buyers

    #########Demand satisfaction and assignment to buyers#############

    translation_demandsatifaction={"DS":1, "UD":0.7, "OD":1.3} # demandsatisfaction: if DS, then demand is satisfied, if UD (Underdemand), then aggregated demand is 70% of supply (oversupply), if OD(overdemand), then demand is 130% of available supply
    totaldemand=translation_demandsatifaction[demandsatisfaction]*totalsupply #calculation of total demand based on the demand satisfaction parameter

    demand_assignment=[] #demand vector for all buyers (the sum of the demands of all buyers is equal to the total demand)
    if demand_variance=="D-AVG": #all buyers have the same demand, which is the average demand
        demand_assignment=[round(totaldemand/QTY_Buyers) for x in range(QTY_Buyers)] #validation not necessary, since the sum of the demands is equal to the total demand
    elif demand_variance=="D-DIST":
        #the demands are normally distributed around the average demand
        mean=totaldemand/QTY_Buyers
        std=math.sqrt(totaldemand/QTY_Buyers)
        
        for x in range(QTY_Buyers):
            demand_assignment.append(round(np.random.normal(mean,std)))
        if sum(demand_assignment)>totaldemand:
            #if the sum of the demands is more than 10% higher than the total demand, then the demands are scaled down
            scale_factor=totaldemand/sum(demand_assignment)
            demand_assignment=[round(demand*scale_factor) for demand in demand_assignment]
    
    else: #demand_variance=="D-RAND":
        volumes = random.choices(range(int(totaldemand) + 1), k=QTY_Buyers)  # randomly choose n values from 0 to total_volume
        scale_factor = totaldemand / sum(volumes)  # calculate the scaling factor to ensure the sum of the volumes equals the total_volume
        demand_assignment = [max(1,round(volume * scale_factor)) for volume in volumes]  # scale the volumes to ensure the sum equals roughly the total_volume
    
    #########Valuations#############
    #The individual valuations are generated as follows: A buyers belongs to one of three groups. Each group has a different valuation scheme.
    #The valuation scheme is chosen randomly for each buyer. The valuation scheme is as follows:
    #1. All items have a random valuation between 1 and the valuation range
    #2. 70% of items have a valuation of 0, the other 30% have a valuation of 75% of the valuation range
    #3. The valuation of an item as a function of the supply of that item.
    for y in range(QTY_Buyers):
        #rc= random choice from 1 to 3
        rc=random.choice(["RANDOM","SELECTIVE","COLLECTOR"])
        valuations={}
        if rc=="RANDOM":
            for k in item_list:
                valuations[k.productID]=random.randint(1,valuations_range)
            
        elif rc=="SELECTIVE":
            # for 70% of items, the valuation is 0, for the others, the valuation is 75% of the range of valuations, rounded up to the nearest integer
            for k in item_list:
                if random.random()<=0.7:
                    valuations[k.productID]=0
                else:
                    valuations[k.productID]=math.ceil(0.75*valuations_range)
                
        elif rc=="COLLECTOR":
            valuations={}
            if min(supply)==max(supply):
                for k in item_list:
                    valuations[k.productID]=int(0.5*valuations_range) #only relevant if all items have the same supply
            else: 
                for k in item_list:
                    #the valuation of an item is a function of the supply of that item- the more supply, the lower the valuation
                    valuations[k.productID]=round(valuations_range-((k.supply-min(supply))/(max(supply)-min(supply)))*valuations_range)
        
        buyer_list.append(buyerinfo("B"+str(y+1),demand_assignment[y],valuations))

    instance=auction_instance(item_list,buyer_list)

    return instance

def transform_tag_to_param(tag):
    #this function takes a tag string and returns the corresponding parameter values
    #the string is in the format |10B|10S|UD|D-AVG|1|S-AVG|#2| with "|" as the delimiter
    #the function returns a list of the parameter values
    params=[]
    
    tag=tag.split("|")
    params.append(int(tag[1][:-1])) #number of buyers
    params.append(int(tag[2][:-1])) #number of items
    params.append(tag[3]) #demand satisfaction
    params.append(tag[4]) #demand variance
    params.append(tag[5]) #range of valuations
    params.append(tag[6]) #supply variance
    params.append(int(tag[7][1:])) #instance number
    

    return params




def generate_data():
    #----configuration of the instances----
    #numbers [mutuable]:
    QTY_Buyers=[10,20,50,60] #buyer count
    QTY_Items=[10,20,50,60] #item count
    val_range=[0.5,1,1.5] #valuation range
    #options [immutable due to the nature of the problem, but could be expanded]:
    demand_coverage=["UD","DS","OD"] #demand satisfaction - underdemand, demand satisfaction, overdemand
    demand_variability=["D-AVG","D-DIST","D-RAND"] #variance of buyer demands
    supply_variability=["S-AVG","S-DIST","S-RAND"] #supply variance
    #The individual valuations are populated randomly, so there is no relevant parameter option.

    #----generate a list of instances----
    instances=[]
    for i in itertools.product(QTY_Buyers,QTY_Items,demand_coverage,demand_variability,val_range,supply_variability):
        for j in range(3):
            k=generate_scenario(i[0],i[1],i[2],i[3],i[4],i[5])
            k.tag="|"+str(i[0])+"B|"+str(i[1])+"S|"+str(i[2])+"|"+str(i[3])+"|"+str(i[4])+"|"+str(i[5])+"|#"+str(j+1)+"|"
            instances.append(k)
            
         

    #generate a list of products and buyers
    products=[]
    buyers=[]
    for i in instances:
        for p in i.products:
            products.append(p)
        for b in i.buyers:
            buyers.append(b)
    return instances,products,buyers

GENERATESCENARIO=False
if GENERATESCENARIO==True:
    instances,products,buyers=generate_data()
    #save the instances to a file
    with open('instances.pickle', 'wb') as f:
        pickle.dump(instances, f)
        print("instances saved")
    
#open the instances from a file
with open('instances.pickle', 'rb') as f:
    instances = pickle.load(f)
    print("instances loaded")

PERFORM_ALGORITHM=False
if PERFORM_ALGORITHM==True:
    print("----Start of the price-raising algorithm for all instances----")
    timearchive=[]
    onlyzeros=0  
    tagsoffree=[]
    results=[]
    for i in instances:
        #count the calculation time
        start_time = time.time()
        price_raising_algorithm(i)
        time_required=time.time() - start_time
        prices=[]

        for j in i.products:
            prices.append(j.price)
            
        results.append([i.tag,transform_tag_to_param(i.tag),prices,time_required])

    #store the results in pickle
    with open('results.pickle', 'wb') as f:
        pickle.dump(results, f)
        print("results saved")

#open the results from the pickle file
with open('results.pickle', 'rb') as f:
    results = pickle.load(f)
    print("results loaded")


print("----End of the price-raising algorithm for all instances----")
print(len(instances))


#calculate the assignment of items to buyers
def calculate_assignment(tag):
    for i in instances:
        if i.tag==tag:
            current_instance=i
            break
    #find the prices for this instance in results
    prices=[]
    for i in results:
        if i[0]==tag:
            prices=i[2]
            print("prices found",prices)
            break   

    
    #modify the current instance with the prices
    #for i in range(len(current_instance.products)):
        #current_instance.products[i].price=prices[i]

    #update auction information
    current_instance.update_auction()
    G,a=current_instance.create_network()
    drawgraph(G)
    #find omega3
    
    for i in current_instance.buyers:
        i.omega3=[]
        for j in current_instance.products:   
            if i.valuations[j.productID]-j.price==0:
                i.omega3.append(j.productID)
    
    #find sum of demands of buyers
    sum_of_demands=0
    for i in current_instance.buyers:
        sum_of_demands+=i.demand
    #find sum of supplies of products
    sum_of_supplies=0
    for i in current_instance.products:
        sum_of_supplies+=i.supply

    difference=abs(sum_of_supplies-sum_of_demands)
    if sum_of_demands>sum_of_supplies:
        #add dummy object with supply=difference
        current_instance.products.append(productinfo("P0",difference))
        G.add_node("P0",subset=2)
        for j in current_instance.buyers:
            j.valuations["P0"]=0

    if sum_of_supplies>sum_of_demands:
        #add dummy buyer with demand=difference
        current_instance.buyers.append(buyerinfo("B0",difference,{}))
        G.add_node("B0",subset=1)
        for j in current_instance.products:
            current_instance.buyers[-1].valuations[j.productID]=0
    
    #create networkx graph using create_network
    G,a=current_instance.create_network()
    #for each buyer, add a node with name i.buyerID+"'''",subset=1
    for i in current_instance.buyers:
        G.add_node(i.buyerID+"'''",subset=1)
        G.add_edge(i.buyerID+"'''",i.buyerID,capacity=i.demand)
    #for each product, add a edge from i.buyerID+"'''" to i.productID ,capacity=i.supply

    for i in current_instance.products:
        G.add_edge(i.buyerID+"'''",i.productID,capacity=i.supply)
    #find the maximum flow
    flow_value, flow_dict = nx.maximum_flow(G, "B0'''", "P0")
    #print(flow_value)

    return current_instance


#transfort the results into a dataframe
df=pd.DataFrame(results,columns=["tag","parameters","prices","time"])
params_df=pd.DataFrame(df["parameters"].to_list(),columns=["QTY_Buyers","QTY_Items","demand_coverage","demand_variability","val_range","supply_variability","instance_number"])
df=df.join(params_df)
df=df.drop(columns=["parameters"],axis=1)
df=df.drop(columns=["tag"],axis=1)
pd.set_option('max_columns', None)
print(df.head())

print("_____________________")
df.drop(columns=["instance_number","prices"],axis=1,inplace=True)
#reduce the dataframe by merging the rows with the same parameters (instance_number is irrelevant) and calculating the average time
df=df.groupby(["QTY_Buyers","QTY_Items","demand_coverage","demand_variability","val_range","supply_variability"]).mean()
df=df.reset_index()
print(df.head())

#plot the results
plot0 = sns.FacetGrid(df, col="QTY_Buyers")
plot0.map_dataframe(sns.barplot,ci=None, x="QTY_Items", y="time",palette="Blues_d",saturation=0.5)
plot0.set_axis_labels("Number of Objects", "Average Time (s)")
plt.subplots_adjust(top=0.8)
#plot0.add_legend(title="Number of Buyers")
#add a title to the figure
plt.suptitle("General statistic")
#plt.show()
plt.savefig("facetgrid0.pdf")



plot1 = sns.FacetGrid(df, col="val_range")
plot1.map_dataframe(sns.barplot, ci=None,x="QTY_Items", y="time", hue="QTY_Buyers",palette="Blues_d",saturation=0.5)
plot1.set_axis_labels("Number of Objects", "Average Time (s)")
plt.subplots_adjust(top=0.8)
plot1.add_legend(title="Number of Buyers")
#add a title to the figure
#plt.suptitle("Demand Coverage: OD")
plot1.savefig("facetgrid1.pdf")
#plt.show()

plot2 = sns.FacetGrid(df, col="demand_coverage",col_order=["UD","DS","OD"])
plot2.map_dataframe(sns.barplot, ci=None,x="QTY_Items", y="time", hue="QTY_Buyers",palette="Blues_d",saturation=0.5)
plot2.set_axis_labels("Number of Objects", "Average Time (s)")
plt.subplots_adjust(top=0.8)
plot2.add_legend(title="Number of Buyers")
#add a title to the figure

#plt.suptitle("Demand Coverage: OD")
plot2.savefig("facetgrid2.pdf")
#plt.show()

plot3 = sns.FacetGrid(df, col="demand_variability")
plot3.map_dataframe(sns.barplot, ci=None,x="QTY_Items", y="time", hue="QTY_Buyers",palette="Blues_d",saturation=0.5)
plot3.set_axis_labels("Number of Objects", "Average Time (s)")
plt.subplots_adjust(top=0.8)
plot3.add_legend(title="Number of Buyers")
plot3.savefig("facetgrid3.pdf")
#plt.show()

plot4 = sns.FacetGrid(df, col="supply_variability")
plot4.map_dataframe(sns.barplot, ci=None,x="QTY_Items", y="time", hue="QTY_Buyers",palette="Blues_d",saturation=0.5)
plot4.set_axis_labels("Number of Objects", "Average Time (s)")
plt.subplots_adjust(top=0.8)
plot4.add_legend(title="Number of Buyers")
plot4.savefig("facetgrid4.pdf")
#plt.show()

#similar to above, create a facetgrid with different parameters for rows and columns
plot5 = sns.FacetGrid(df, col="demand_coverage",row="demand_variability",col_order=["UD","DS","OD"])
plot5.map_dataframe(sns.barplot, ci=None,x="QTY_Items", y="time", hue="QTY_Buyers",palette="Blues_d",saturation=0.5)
plot5.set_titles(" {row_name},{col_name}")
plot5.set_axis_labels("Number of Objects", "Average Time (s)")
plt.subplots_adjust(top=0.8)
plot5.add_legend(title="Number of Buyers")
plot5.savefig("facetgrid5.pdf")
plt.show()

plot6 = sns.FacetGrid(df, col="demand_coverage",row="supply_variability",col_order=["UD","DS","OD"])
plot6.map_dataframe(sns.barplot, ci=None,x="QTY_Items", y="time", hue="QTY_Buyers",palette="Blues_d",saturation=0.5)
plot6.set_axis_labels("Number of Objects", "Average Time (s)")
plot6.set_titles(" {row_name},{col_name}")
plt.subplots_adjust(top=0.8)
plot6.add_legend(title="Number of Buyers")
plot6.savefig("facetgrid6.pdf")
plt.show()

plot7 = sns.FacetGrid(df, col="demand_variability",row="supply_variability")
plot7.map_dataframe(sns.barplot, ci=None,x="QTY_Items", y="time", hue="QTY_Buyers",palette="Blues_d",saturation=0.5)
plot7.set_axis_labels("Number of Objects", "Average Time (s)")
plot7.set_titles(" {row_name},{col_name}")
plt.subplots_adjust(top=0.8)
plot7.add_legend(title="Number of Buyers")
plot7.savefig("facetgrid7.pdf")
plt.show()

plot8 = sns.FacetGrid(df, col="demand_coverage",row="val_range",col_order=["UD","DS","OD"])
plot8.map_dataframe(sns.barplot, ci=None,x="QTY_Items", y="time", hue="QTY_Buyers",palette="Blues_d",saturation=0.5)
plot8.set_axis_labels("Number of Objects", "Average Time (s)")
plot8.set_titles(" {row_name},{col_name}")
plt.subplots_adjust(top=0.8)
plot8.add_legend(title="Number of Buyers")
plot8.savefig("facetgrid8.pdf")
plt.show()

plot9= sns.FacetGrid(df, col="demand_variability",row="val_range")
plot9.map_dataframe(sns.barplot, ci=None,x="QTY_Items", y="time", hue="QTY_Buyers",palette="Blues_d",saturation=0.5)    
plot9.set_axis_labels("Number of Objects", "Average Time (s)")
plot9.set_titles(" {row_name},{col_name}")
plt.subplots_adjust(top=0.8)
plot9.add_legend(title="Number of Buyers")
plot9.savefig("facetgrid9.pdf")

plot10= sns.FacetGrid(df, col="supply_variability",row="val_range")
plot10.map_dataframe(sns.barplot, ci=None,x="QTY_Items", y="time", hue="QTY_Buyers",palette="Blues_d",saturation=0.5)
plot10.set_axis_labels("Number of Objects", "Average Time (s)")
plot10.set_titles(" {row_name},{col_name}")
plt.subplots_adjust(top=0.8)
plot10.add_legend(title="Number of Buyers")
plot10.savefig("facetgrid10.pdf")



def monotony_analysis(instances,results,tag):
    #for a given tag, create a copy c of the instance and change the parameters:
    # for c, increase supply
    # then run the price-raising algorithm on c 
    #at last compare the results of the price-raising algorithm on the original instance and on c   
    test_supply=True
    test_demand=False
     
    
    #find the instance with the given tag
    for i in instances:
        if i.tag==tag:
            current_instance=i
            break
    #find the prices for this instance in results
    prices=[]
    for i in results:
        if i[0]==tag:
            prices=i[2]
            #print("prices found",prices)
            break
    if test_supply:
    #increase supply for current_instance
        for i in current_instance.products:
            i.supply=math.ceil(i.supply*10)
    if test_demand:
        for i in current_instance.buyers:
            i.demand=math.ceil(i.demand*10)

    #update auction information
    current_instance.update_auction()
    price_raising_algorithm(current_instance)

    #find the prices for the new instance
    prices_new=[]
    for i in current_instance.products:
        prices_new.append(i.price)
     

    #compare the prices, find out if all new prices are lower or equal to the old prices
    all_lower=True
    if test_supply:
        for i in range(len(prices)):
            if prices_new[i]>prices[i]:
                all_lower=False
                break

        if all_lower:
            print("all prices are lower or equal")
        else:
            print("not all prices are lower or equal")
    if test_demand:
        all_higher=True
        for i in range(len(prices)):
            if prices_new[i]<prices[i]:
                all_higher=False
                break

        if all_lower:
            print("all prices are higher or equal")
        else:
            print("not all prices are higher or equal")
    return
    

    
#for tags in [instances[i].tag for i in range(len(instances))]:
    #monotony_analysis(instances,results,tags)
