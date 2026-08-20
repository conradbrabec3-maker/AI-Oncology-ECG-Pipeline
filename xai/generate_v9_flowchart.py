import base64
import requests
import sys

def main():
    # 1. Define the V9 Architecture using Mermaid.js syntax
    mermaid_code = """
    graph TD
        %% Global Input
        Global_Input([Raw 12-lead ECG Waveform Input])

        %% Subgraph 1: Branch A
        subgraph Branch_A [Branch A: Structural Feature Extractor]
            direction TB
            CNN["Deep 1D-CNN<br>(Batch Normalization & ReLU)"]
            Features_A["Localized Spatial-Temporal Features"]
        end

        %% Subgraph 2: SE Attention
        subgraph SE_Attention [Squeeze-and-Excitation Attention]
            direction TB
            SE_Blocks{"SE Attention Blocks<br>(Mutes ICU stress, Amplifies oncological signatures)"}
            Flatten["Flatten to High-Dimensional Vector"]
            Bottleneck["Linear Bottleneck Layer<br>(Compresses data)"]
        end

        %% Subgraph 3: Branch B
        subgraph Branch_B [Branch B: Explicit HRV Timing Branch]
            direction TB
            Lead_II["Lead II ONLY"]
            Peak_Detect["SciPy Peak Detection Process"]
            HRV_Metrics[/"Explicit Tabular HRV Metrics<br>(Mean N-N, SDNN, RMSSD)"/]
        end

        %% Subgraph 4: Multimodal Fusion & Classification Head
        subgraph Fusion_Head [Multimodal Fusion & Classification Head]
            direction TB
            Concat{"Concatenation Layer"}
            Dense["Dense nn.Linear Layer(s) with Dropout"]
            Sigmoid("Logistic Sigmoid Activation")
            Final_Output([Final Binary Probability Distribution<br>Cancer vs. Healthy Control])
        end

        %% Define the flow/connections
        Global_Input --> CNN
        CNN --> Features_A
        Features_A --> SE_Blocks
        SE_Blocks --> Flatten
        Flatten --> Bottleneck

        Global_Input --> Lead_II
        Lead_II --> Peak_Detect
        Peak_Detect --> HRV_Metrics

        Bottleneck --> Concat
        HRV_Metrics --> Concat
        
        Concat --> Dense
        Dense --> Sigmoid
        Sigmoid --> Final_Output
        
        %% Styling to make it look professional
        classDef primary fill:#e1f5fe,stroke:#333,stroke-width:2px;
        classDef process fill:#fff3e0,stroke:#333,stroke-width:2px;
        classDef attention fill:#fce4ec,stroke:#333,stroke-width:2px;
        classDef output fill:#e8f5e9,stroke:#333,stroke-width:2px;
        
        class Global_Input,Lead_II primary;
        class CNN,Peak_Detect,Flatten,Bottleneck,Dense process;
        class SE_Blocks,Concat attention;
        class Final_Output output;
    """

    print("⏳ Encoding Mermaid.js graph...")
    # 2. Encode the string to Base64 (URL Safe) so the API can read it
    encoded_graph = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
    
    # 3. Construct the API request (requesting a white background for clean saving)
    api_url = f"https://mermaid.ink/img/{encoded_graph}?bgColor=white"
    
    print("🚀 Requesting image render from Mermaid.ink API...")
    try:
        response = requests.get(api_url, timeout=15)
        
        if response.status_code == 200:
            output_filename = "v9_architecture.png"
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ Success! Flowchart securely saved to ARC storage as '{output_filename}'")
        else:
            print(f"❌ Error {response.status_code}: Could not generate image.")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: Could not reach the API. {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
