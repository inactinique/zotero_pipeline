import spacy
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import pandas as pd
from IPython.display import display, HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class NERProcessor:
    def __init__(self, model="en_core_web_sm"):
        """
        Initialize the NER processor
        
        Args:
            model (str): spaCy model to use (e.g., "en_core_web_sm", "fr_core_news_sm")
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Downloading language model {model}...")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)
        
        # Standard entity colors from spaCy
        self.colors = {
            'PERSON': '#7aecec',
            'ORG': '#ffd966',
            'GPE': '#ff9966',
            'LOC': '#ff9966',
            'DATE': '#bfe1d9',
            'TIME': '#bfe1d9',
            'MONEY': '#e4e7d2',
            'PERCENT': '#e4e7d2',
            'WORK_OF_ART': '#ffeb80',
            'LAW': '#ff9966',
            'LANGUAGE': '#ff9966',
            'EVENT': '#bfeeb7',
            'FACILITY': '#ffd966',
            'PRODUCT': '#bfeeb7'
        }

    def process_texts(self, texts: List[str], metadata: List[Dict]) -> List[Dict]:
        """
        Process multiple texts and extract named entities
        """
        results = []
        for text, meta in zip(texts, metadata):
            doc = self.nlp(text)
            
            # Extract entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Count entity types
            entity_counts = Counter(label for _, label in entities)
            
            # Group entities by type
            entities_by_type = defaultdict(list)
            for text, label in entities:
                entities_by_type[label].append(text)
            
            # Create result dictionary
            result = {
                'title': meta.get('title', 'Untitled'),
                'authors': '; '.join(author.get('name', '') for author in meta.get('creators', [])),
                'date': meta.get('date', ''),
                'entity_counts': dict(entity_counts),
                'entities_by_type': dict(entities_by_type),
                'all_entities': entities
            }
            results.append(result)
        
        return results

    def display_entity_summary(self, results: List[Dict]):
        """
        Display summary of entities across all documents
        """
        # Aggregate counts across all documents
        total_counts = Counter()
        for result in results:
            total_counts.update(result['entity_counts'])
        
        # Create summary DataFrame
        summary_data = pd.DataFrame([
            {'Entity Type': ent_type, 'Count': count}
            for ent_type, count in total_counts.most_common()
        ])
        
        # Create bar chart
        fig = px.bar(summary_data, 
                    x='Entity Type', 
                    y='Count',
                    color='Entity Type',
                    color_discrete_map={k: v for k, v in self.colors.items() if k in total_counts},
                    title='Named Entities Distribution Across All Documents')
        
        fig.update_layout(showlegend=False)
        fig.show()
        
        # Display detailed table
        styled_df = summary_data.style\
            .set_properties(**{'border': '1px solid black',
                             'padding': '8px',
                             'text-align': 'left'})\
            .set_table_styles([{'selector': 'th',
                               'props': [('border', '1px solid black'),
                                       ('padding', '8px'),
                                       ('text-align', 'left')]}])
        
        display(HTML("<h3>Entity Type Counts</h3>"))
        display(styled_df)

    def display_document_entities(self, results: List[Dict]):
        """
        Display entities found in each document
        """
        for result in results:
            # Create HTML header for document
            header = f"""
            <h3>Document: {result['title']}</h3>
            <p><strong>Authors:</strong> {result['authors']}</p>
            <p><strong>Date:</strong> {result['date']}</p>
            """
            display(HTML(header))
            
            # Create entities table
            entities_data = []
            for ent_type, entities in result['entities_by_type'].items():
                # Get unique entities with their counts
                entity_counts = Counter(entities)
                entities_str = ", ".join(f"{entity} ({count})" 
                                       for entity, count in entity_counts.most_common())
                
                entities_data.append({
                    'Entity Type': ent_type,
                    'Count': len(entities),
                    'Unique Entities': entities_str
                })
            
            df = pd.DataFrame(entities_data)
            
            # Style and display the table
            styled_df = df.style\
                .set_properties(**{'border': '1px solid black',
                                 'padding': '8px',
                                 'text-align': 'left'})\
                .set_table_styles([{'selector': 'th',
                                   'props': [('border', '1px solid black'),
                                           ('padding', '8px'),
                                           ('text-align', 'left')]}])
            
            display(styled_df)
            display(HTML("<hr>"))

    def create_entity_network(self, results: List[Dict]):
        """
        Create and display an entity co-occurrence network
        """
        import networkx as nx
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes and edges
        for result in results:
            # Group entities by sentence
            doc = self.nlp(result['title'])  # Using title as a minimal example
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Add nodes
            for entity, ent_type in entities:
                if not G.has_node(entity):
                    G.add_node(entity, type=ent_type)
            
            # Add edges between co-occurring entities
            entities_list = [ent for ent, _ in entities]
            for i, ent1 in enumerate(entities_list):
                for ent2 in entities_list[i+1:]:
                    if G.has_edge(ent1, ent2):
                        G[ent1][ent2]['weight'] += 1
                    else:
                        G.add_edge(ent1, ent2, weight=1)
        
        # Create visualization using plotly
        pos = nx.spring_layout(G)
        
        # Create edges trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create nodes trace
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_colors.append(self.colors.get(G.nodes[node]['type'], '#888'))
            node_text.append(f"{node}<br>{G.nodes[node]['type']}")
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_colors,
                size=10,
                line_width=2))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Entity Co-occurrence Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0,l=0,r=0,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        fig.show()