- Switch 
    
    - Device we use to enable computers in same environment to communicate with each other
        
    - Connected with computers through a wire (not wireless)
        
    - CAT-5 or 6 cable in small env
        
    - Switch + Computers = Network -> LAN (Local area network)
- Access point device
    
    - Wireless switch basically
- Packets/Frames
    - Information sent by one computer to other
    - Packet -> Switch -> Destination computer
- LAN Ports
    
    - Ports in switch and computers to connect to local area network
- Router
    
    - Device we use to connect to the internet
        
    - Connected with a wire that goes outside the environment to ISP (internet service provider, paid service)
        
    - Home Router
        
    - Router + Switch, good for small environments
        
    - If env is big then switch needs to be separate
- Internet
    
	- Networks of networks
    
	- Network that connects all the LANs to each other that are connected to it
- Single point failure
    
    - If one device fails then the system fails
        
    - The internet has distributed architecture to avoid single point failure
- Forwarding
    - Information passed on by router after learning its destination
- Routing tables
    
    - are created by special processors using special algorithms to decide which router to send packet to
        
    - Generated for "Congestion control"
        
        - If a router is busy/overloaded then it can be sent to some alternate path
- Server
    - Special computers that are more computationally powerful as compared to client device as they have to send a lot of data
    - Distributed server avoids single point failure and does load balancing
- WAN (wide area network)
    
    - Internet is public, whereas WAN is private network although it spans over long distance
        
    - Costly unless done through "VPN"
        
        - Virtual Private Network
            
        - Uses tunneling which creates a "tunnel" or a special connection within public network
            
    - ISP WAN
        
        - Special wire laid by the ISP company to create a private WAN
- Encryption
    
    - End-to-End: Information encrypted at client side and decrypted at server side
- ISP
    - Organised into hierarchal structure
        
        - Global
            
        - Regional
            
        - Local
            
            - In small offices called "POP: Point of presence"
    - Peering
        - Establishing direct connection to Local ISP and lower levels skipping long paths
            
        - Distributed Server + Peering = High volume internet products like google amazon etc    
    - Internet exchange point (IXPs)
        - Structure that allows internet backbone (Global ISPs) to work together
- BLA BLA