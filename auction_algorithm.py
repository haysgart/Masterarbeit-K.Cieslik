# -*- coding: utf-8 -*-
import pprint as pp
import networkx as nx
import time
import matplotlib.pyplot as plt
from networkx.algorithms.flow import edmonds_karp
import graphviz

class productinfo:
    """class to store information about a product"""
    def __init__(self, productID, supply,price=0):
        """productID is a string, supply is an integer"""
        self.productID=productID
        self.supply=supply
        self.price=0 #inital price is zero, will be updated in the auction

    #function to increment the price ot the product by one
    def incrementprice(self):
        """increment the price of the product by one"""
        self.price=self.price+1

class buyerinfo:
    """class to store information about a buyer"""
    def __init__(self, buyerID, demand,valuations):
        """buyerID is a string ("b" followed by number),demand is an integer, valuations is a dictionary with productID as key and valuation as value"""
        self.buyerID=buyerID 
        self.demand=demand #demand is the total amount of products the buyer wants to buy, and can be used for any product
        self.valuations=valuations #dictionary with productID as key and valuation as value

class auction_instance:
    """class to store information about an auction instance"""
    def __init__(self, products, buyers): 
        self.products=products
        self.buyers=buyers
        self.sortedpayoffs=self.calc_sortedpayoffs() #dictionary with buyerID as key and list of lists of productIDs as value
        self.omega1,self.omega2=self.calc_omegas() #dictionary with buyerID as key and list of productIDs as value
        self.tag=""
    
    def update_auction(self):
        """update the auction instance after a round of the auction"""
        self.sortedpayoffs=self.calc_sortedpayoffs()
        self.omega1,self.omega2=self.calc_omegas()
        
    
    def calc_sortedpayoffs(self):
        """calculate the sortedpayoffs dictionary"""
        sortedpayoffs={}
        for i in self.buyers:
            #find all unique possible positive payoffs (payoff= valuation minus price) and sort them
            payoffs=list(set([i.valuations[j.productID]-j.price for j in self.products if i.valuations[j.productID]-j.price>0]))
            payoffs.sort(reverse=True)
            #for all possible payoffs, find the set of products that give that payoff
            sortedpayoffs[i.buyerID]=[]
            for k in payoffs:
                sortedpayoffs[i.buyerID].append([j.productID for j in self.products if i.valuations[j.productID]-j.price==k])
        
        return sortedpayoffs
    

    def calc_omegas(self):
        """calculate the omega1 and omega2 dictionaries"""
        omega1={} #contains all products per buyer that have the highest payoff and their total supply is leq the demand of the buyer(--> can be bought as complete package)
        omega2={} #contains all products per buyer that have the highest payoff and their total supply is greater than the demand of the buyer(--> can only be bought partially)

        #iterate over all buyers and lists in sortedpayoffs
        for buyer in self.buyers:
            n=0
            key=buyer.buyerID
            omega1[key]=[]
            omega2[key]=[]
            highestvalue=0

            remainingdemand=buyer.demand
            while remainingdemand>0 and n<len(self.sortedpayoffs[key]):
                #1. find the highest value in the sortedpayoffs list
                #2. find the total supply of the products with that value
                #3. if the total supply is greater than the remaining demand, add all products with that value to omega2
                #4. if the total supply is equal to the remaining demand, add all products with that value to omega1
                #5. if the total supply is less than the remaining demand, add all products with that value to omega1 and update the remaining demand
                #6. repeat steps 1-5 until remaining demand is zero or all products are assigned to omega1 or omega2
                highestvalue=self.sortedpayoffs[key][n]
                
                totalsupply=sum([self.products[self.products.index(j)].supply for j in self.products if j.productID in highestvalue])
               
                if totalsupply>remainingdemand:
                    omega2[key].extend(highestvalue)
                    remainingdemand=0
                elif totalsupply==remainingdemand:
                    omega1[key].extend(highestvalue)
                    remainingdemand=0
                else:
                    omega1[key].extend(highestvalue)
                    remainingdemand=remainingdemand-totalsupply
                    n=n+1
        return omega1, omega2

    def create_network(self):
        """create a networkx graph with the nodes and edges of the auction instance"""

        network = nx.DiGraph()
        #add nodes: one source s, two nodes per buyer, one node for each product, one sink t
        #For each buyer i, we add two nodes: i_1 and i_2. i_1 is in omega1 and i_2 is in omega2.
        #The duplicate nodes are needed to distinguish between the two sets of products that can be bought by the buyer.

        network.add_node("s",subset=0)
        for i in self.buyers:
            network.add_node(i.buyerID+"'",subset=1)
            network.add_node(i.buyerID+"''",subset=1)

        for i in self.products:
            network.add_node(i.productID,subset=2)

        network.add_node("t",subset=3)

        #add edges
        #add edges from source to buyers
        flowconservation_totaldemand=0 #used in the price raising algorithm to check if the flow conservation constraint is satisfied
        for i in self.buyers:
            #add edges from source to buyer nodes in omega1
            hj1=sum([self.products[self.products.index(j)].supply for j in self.products if j.productID in self.omega1[i.buyerID]])
            #hj1, as in the paper, is the total supply of the products purchasable in omega1 for buyer i

            if len(self.omega1[i.buyerID])>0:
                network.add_edge("s",i.buyerID+"'",capacity=hj1)
            flowconservation_totaldemand=flowconservation_totaldemand+hj1 
            
            hj2=min(sum([self.products[self.products.index(j)].supply for j in self.products if j.productID in self.omega2[i.buyerID]]),i.demand-hj1)
            #hj2, as in the paper, is the total supply of the products purchasable in omega2 for buyer i.
            #It is the minimum of the total supply of the products purchasable in omega2 and the remaining demand of buyer i after the products in omega1 have been bought.
            if len(self.omega2[i.buyerID])>0:
                hj2=min(sum([self.products[self.products.index(j)].supply for j in self.products if j.productID in self.omega2[i.buyerID]]),i.demand-hj1)
                network.add_edge("s",i.buyerID+"''",capacity=hj2)
            flowconservation_totaldemand=flowconservation_totaldemand+hj2
            
            

        #add edges from buyers to products
        for i in self.buyers:
            for j in self.products:
                hj1=sum([self.products[self.products.index(j)].supply for j in self.products if j.productID in self.omega1[i.buyerID]])
                hj2=min(sum([self.products[self.products.index(j)].supply for j in self.products if j.productID in self.omega2[i.buyerID]]),i.demand-hj1)
                #hj1 and hj2 are as explained above
                gij1= j.supply
                #gij1 is the supply of product j
                if j.productID in self.omega1[i.buyerID]:
                    network.add_edge(i.buyerID+"'",j.productID,capacity=gij1)


                gij2= min(j.supply,hj2)
                #gij2 is either the supply of product j or the remaining total demand of buyer i for items in omega2 after the products in omega1 have been bought, whichever is smaller.
                #in other words, the buyer wants to buy at most hj2 units of product j, but the supply of product j is gij2, which might be lower than hj2.
                if j.productID in self.omega2[i.buyerID]:
                    network.add_edge(i.buyerID+"''",j.productID,capacity=gij2)
            
               
        #add edges from products to sink
        for i in self.products:
            network.add_edge(i.productID,"t",capacity=i.supply)

        
        return network, flowconservation_totaldemand


def drawgraph(network):
    pos = nx.multipartite_layout(network, subset_key="subset")
    nx.draw(network, pos,node_color="cornflowerblue", with_labels=True,arrows=True,node_size=800)
    labels = nx.get_edge_attributes(network, 'capacity')
    nx.draw_networkx_edges(network, pos,arrows=False,edgelist=labels.keys(), alpha=1, min_source_margin=20, min_target_margin=20)
    
    nx.draw_networkx_edge_labels(network,pos,label_pos=0.7,edge_labels=labels, font_size=7, alpha=1,rotate="false")
    #plt.show()
    

def findmaxflow(network):
    #perform max flow algorithm
    flow_value, flow_dict = nx.maximum_flow(network, "s", "t")

    return flow_value, flow_dict


def findresidual(network):
    """based on the max flow algorithm, find the residual network after the final flow has been found"""
    fv,fd=findmaxflow(network)
    residual=nx.DiGraph()
    for node in network.nodes:
        residual.add_node(node,subset=network.nodes[node]["subset"])
    for node in residual.nodes:
        for v in network[node]: 
            residual_capacity=network[node][v]["capacity"]-fd[node][v] #capacity is original capacity minus the flow
            if residual_capacity>0:
                residual.add_edge(node,v,capacity=residual_capacity)
    
    return residual

def findleftmostmincut(network):
    """find the leftmost minimum cut"""
    resi=findresidual(network)
    reachable_nodes=set(nx.algorithms.traversal.bfs_tree(resi,"s").nodes())
    left_partition=[node for node in network.nodes if node in reachable_nodes]
    
    draw_cut(network,left_partition)
    return left_partition

def draw_cut(network, cut_set):
    subgraph_nodes = set(list(cut_set)+["s","t"]+list(network.nodes()))
   
    subgraph = network.subgraph(subgraph_nodes)
    pos = nx.multipartite_layout(subgraph, subset_key="subset")
    # Draw the subgraph
    #nx.draw(subgraph, pos, node_color=['tomato' if node in cut_set else 'dodgerblue' for node in subgraph.nodes()], node_size=1400,  with_labels=True,font_size=16)
    #plt.show()

def price_raising_algorithm(auction):
    
    iter=1 #iteration counter   
    #print("Starting price raising algorithm")
    network, sumofdemands= auction.create_network()
    #drawgraph(network)
    flow_value,flow_details=findmaxflow(network)
   

    while flow_value<sumofdemands:
        
        auction.update_auction()

        network, sumofdemands= auction.create_network()
        cut_set= findleftmostmincut(network)
        #draw_cut(network, cut_set)
        for i in auction.products:
           if i.productID in cut_set:
                i.incrementprice()
              
        flow_value,flow_details=findmaxflow(network)
        iter=iter+1
        if iter>=400:
            print("Error: Iteration limit reached!!!!!!!!!!!!!!!")
            #calculate the sum of supplies
            sumofsupplies=0
            for i in auction.products:
                sumofsupplies=sumofsupplies+i.supply
            print("Sum of supplies is ", sumofsupplies)
            print("Flow value is ", flow_value)
            print("Sum of demands is ", sumofdemands)
            print("Flow details are ")
            pp.pprint(flow_details)
            print("Cut set is ", cut_set)
            print("Iteration is ", iter)
            print(auction.tag)
            #print prices
            for i in auction.products:
                print("Price of ", i.productID, " is ", i.price)
            for b in auction.buyers:
                print("Demand of ", b.buyerID, " is ", b.demand)
            
            #print preferences of buyer
            for i in auction.buyers:
                print("Preferences of ", i.buyerID, " are ", i.valuations)

            break
            
                 
    finaliter=iter
    print("Done ")
    #f
    
    return finaliter-1, flow_value, sumofdemands,flow_details
    




