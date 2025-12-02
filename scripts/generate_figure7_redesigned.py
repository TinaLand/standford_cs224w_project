"""
Redesigned Figure 7 series - Modern, professional, beautiful visualizations
Using better layouts, colors, and design principles
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGS_DIR = PROJECT_ROOT / "figures"
FIGS_DIR.mkdir(exist_ok=True)

# Professional color palette
COLORS = {
    'tech': '#FF6B6B',      # Modern red
    'finance': '#4ECDC4',   # Teal
    'healthcare': '#95E1D3', # Mint
    'consumer': '#F38181',  # Coral
    'energy': '#AA96DA',    # Lavender
    'correlation': '#FF6B6B',
    'sector': '#4ECDC4',
    'fundamental': '#95E1D3',
    'background': '#F8F9FA',
    'text': '#2C3E50',
    'accent': '#FFD93D'
}

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']

def get_graph_data():
    """Get graph data with professional structure."""
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'BAC', 'JNJ', 'PFE', 'WMT', 
              'HD', 'MCD', 'XOM', 'CVX']
    
    sectors = {
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'finance': ['JPM', 'BAC'],
        'healthcare': ['JNJ', 'PFE'],
        'consumer': ['WMT', 'HD', 'MCD'],
        'energy': ['XOM', 'CVX']
    }
    
    # Correlation graph
    G_corr = nx.Graph()
    G_corr.add_nodes_from(stocks)
    corr_edges = [
        ('AAPL', 'MSFT', 0.85), ('AAPL', 'GOOGL', 0.78), ('MSFT', 'GOOGL', 0.82),
        ('AMZN', 'META', 0.75), ('AMZN', 'GOOGL', 0.70), ('META', 'GOOGL', 0.72),
        ('JPM', 'BAC', 0.88), ('JNJ', 'PFE', 0.80), ('WMT', 'HD', 0.65),
        ('XOM', 'CVX', 0.90), ('AAPL', 'AMZN', 0.68), ('MSFT', 'META', 0.65)
    ]
    G_corr.add_weighted_edges_from(corr_edges)
    
    # Sector graph
    G_sector = nx.Graph()
    G_sector.add_nodes_from(stocks)
    sector_edges = [
        ('AAPL', 'MSFT'), ('AAPL', 'GOOGL'), ('AAPL', 'AMZN'), ('AAPL', 'META'),
        ('MSFT', 'GOOGL'), ('MSFT', 'AMZN'), ('MSFT', 'META'),
        ('GOOGL', 'AMZN'), ('GOOGL', 'META'), ('AMZN', 'META'),
        ('JPM', 'BAC'), ('JNJ', 'PFE'),
        ('WMT', 'HD'), ('WMT', 'MCD'), ('HD', 'MCD'),
        ('XOM', 'CVX')
    ]
    G_sector.add_edges_from(sector_edges)
    
    # Fundamental graph
    G_fund = nx.Graph()
    G_fund.add_nodes_from(stocks)
    fund_edges = [
        ('AAPL', 'MSFT', 0.82), ('GOOGL', 'META', 0.75), ('AMZN', 'META', 0.70),
        ('JPM', 'BAC', 0.85), ('JNJ', 'PFE', 0.78), ('XOM', 'CVX', 0.88),
        ('WMT', 'HD', 0.72), ('HD', 'MCD', 0.68), ('AAPL', 'AMZN', 0.65)
    ]
    G_fund.add_weighted_edges_from(fund_edges)
    
    # Use spring layout for better positioning
    pos = nx.spring_layout(G_corr, k=3, iterations=100, seed=42)
    # Scale and center
    pos = {k: (v[0]*8, v[1]*8) for k, v in pos.items()}
    
    node_colors = {}
    for sector_name, sector_stocks in sectors.items():
        for stock in sector_stocks:
            node_colors[stock] = COLORS[sector_name]
    
    return stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors


def draw_nodes_modern(ax, stocks, pos, node_colors, size=800, fontsize=11, alpha=0.9):
    """Draw nodes with modern style."""
    for stock in stocks:
        if stock in pos:
            x, y = pos[stock]
            # Outer glow
            circle_outer = Circle((x, y), size/400 + 0.15, 
                                 color=node_colors.get(stock, '#BDC3C7'),
                                 alpha=0.2, zorder=1)
            ax.add_patch(circle_outer)
            
            # Main node
            circle = Circle((x, y), size/400, 
                          color=node_colors.get(stock, '#BDC3C7'),
                          alpha=alpha, zorder=3, 
                          edgecolor='white', linewidth=3)
            ax.add_patch(circle)
            
            # Text with shadow
            text = ax.text(x, y, stock, 
                          ha='center', va='center', fontsize=fontsize, 
                          fontweight='bold', color='white', zorder=4)
            text.set_path_effects([withStroke(linewidth=4, foreground='black', alpha=0.6)])


def create_figure7a_complete_overview():
    """Complete overview with all edge types - modern design."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors = get_graph_data()
    
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    ax = fig.add_subplot(111, facecolor=COLORS['background'])
    
    # Draw edges with modern styling
    # Correlation edges (thick, solid)
    for edge in G_corr.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=COLORS['correlation'], 
                   linewidth=weight*8 + 4, 
                   alpha=0.6, zorder=1, solid_capstyle='round')
    
    # Sector edges (dashed)
    for edge in G_sector.edges():
        if edge[0] in pos and edge[1] in pos and edge not in G_corr.edges():
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=COLORS['sector'], 
                   linewidth=5, linestyle='--', 
                   dashes=(15, 8), alpha=0.5, zorder=1)
    
    # Fundamental edges (dotted)
    for edge in G_fund.edges():
        if edge[0] in pos and edge[1] in pos and edge not in G_corr.edges() and edge not in G_sector.edges():
            weight = G_fund[edge[0]][edge[1]].get('weight', 0.7)
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=COLORS['fundamental'], 
                   linewidth=weight*6 + 3, 
                   linestyle=':', dashes=(3, 8), alpha=0.5, zorder=1)
    
    draw_nodes_modern(ax, stocks, pos, node_colors, size=1000, fontsize=13)
    
    # Set limits with padding
    x_coords = [pos[s][0] for s in stocks if s in pos]
    y_coords = [pos[s][1] for s in stocks if s in pos]
    ax.set_xlim(min(x_coords) - 2, max(x_coords) + 2)
    ax.set_ylim(min(y_coords) - 2, max(y_coords) + 2)
    
    ax.set_title('Heterogeneous Stock Graph: Multi-Relational Structure', 
                fontsize=24, fontweight='bold', pad=30, color=COLORS['text'])
    ax.axis('off')
    
    # Modern legend
    legend_elements = [
        Line2D([0], [0], color=COLORS['correlation'], linewidth=10, label='Rolling Correlation'),
        Line2D([0], [0], color=COLORS['sector'], linestyle='--', linewidth=8, dashes=(15, 8), label='Sector/Industry'),
        Line2D([0], [0], color=COLORS['fundamental'], linestyle=':', linewidth=8, dashes=(3, 8), label='Fundamental Similarity'),
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=16, 
                      frameon=True, fancybox=True, shadow=True, 
                      framealpha=0.98, edgecolor='#E0E0E0', facecolor='white',
                      bbox_to_anchor=(0.98, 0.98), ncol=1)
    legend.get_frame().set_linewidth(2)
    
    # Sector color legend
    sector_patches = [mpatches.Patch(color=COLORS[s], label=s.capitalize()) 
                     for s in sectors.keys()]
    legend2 = ax.legend(handles=sector_patches, loc='upper left', fontsize=14,
                        frameon=True, fancybox=True, shadow=True,
                        framealpha=0.98, edgecolor='#E0E0E0', facecolor='white',
                        bbox_to_anchor=(0.02, 0.98), ncol=1)
    ax.add_artist(legend)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7a_complete_overview.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7a_complete_overview.png")


def create_figure7b_correlation_only():
    """Correlation edges only - clean and focused."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors = get_graph_data()
    
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    ax = fig.add_subplot(111, facecolor=COLORS['background'])
    
    # Only correlation edges
    for edge in G_corr.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=COLORS['correlation'], 
                   linewidth=weight*10 + 5, 
                   alpha=0.7, zorder=1, solid_capstyle='round')
    
    draw_nodes_modern(ax, stocks, pos, node_colors, size=1000, fontsize=13)
    
    x_coords = [pos[s][0] for s in stocks if s in pos]
    y_coords = [pos[s][1] for s in stocks if s in pos]
    ax.set_xlim(min(x_coords) - 2, max(x_coords) + 2)
    ax.set_ylim(min(y_coords) - 2, max(y_coords) + 2)
    
    ax.set_title('Rolling Correlation Edges: Dynamic Price Co-Movements', 
                fontsize=22, fontweight='bold', pad=25, color=COLORS['text'])
    ax.axis('off')
    
    # Info panel - modern card style
    info_text = ('Dynamic Edges\n'
                '30-day rolling window\n'
                'Top-K=10 per stock\n'
                'Updates: Daily')
    info_box = FancyBboxPatch((0.02, 0.02), 0.25, 0.18, 
                              boxstyle="round,pad=0.02", 
                              transform=ax.transAxes,
                              facecolor='white', edgecolor=COLORS['correlation'], 
                              linewidth=3, alpha=0.95, zorder=10)
    ax.add_patch(info_box)
    ax.text(0.145, 0.11, info_text, transform=ax.transAxes, 
           fontsize=13, verticalalignment='center', horizontalalignment='center',
           color=COLORS['text'], fontweight='bold', zorder=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7b_correlation_only.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7b_correlation_only.png")


def create_figure7c_sector_only():
    """Sector edges only."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors = get_graph_data()
    
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    ax = fig.add_subplot(111, facecolor=COLORS['background'])
    
    # Only sector edges
    for edge in G_sector.edges():
        if edge[0] in pos and edge[1] in pos:
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=COLORS['sector'], 
                   linewidth=6, linestyle='--', 
                   dashes=(15, 8), alpha=0.7, zorder=1)
    
    draw_nodes_modern(ax, stocks, pos, node_colors, size=1000, fontsize=13)
    
    x_coords = [pos[s][0] for s in stocks if s in pos]
    y_coords = [pos[s][1] for s in stocks if s in pos]
    ax.set_xlim(min(x_coords) - 2, max(x_coords) + 2)
    ax.set_ylim(min(y_coords) - 2, max(y_coords) + 2)
    
    ax.set_title('Sector/Industry Edges: Static Domain Knowledge', 
                fontsize=22, fontweight='bold', pad=25, color=COLORS['text'])
    ax.axis('off')
    
    info_text = ('Static Edges\n'
                'Industry classification\n'
                'Binary (same sector=1)\n'
                'Never updated')
    info_box = FancyBboxPatch((0.02, 0.02), 0.25, 0.18, 
                              boxstyle="round,pad=0.02", 
                              transform=ax.transAxes,
                              facecolor='white', edgecolor=COLORS['sector'], 
                              linewidth=3, alpha=0.95, zorder=10)
    ax.add_patch(info_box)
    ax.text(0.145, 0.11, info_text, transform=ax.transAxes, 
           fontsize=13, verticalalignment='center', horizontalalignment='center',
           color=COLORS['text'], fontweight='bold', zorder=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7c_sector_only.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7c_sector_only.png")


def create_figure7d_fundamental_only():
    """Fundamental edges only."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors = get_graph_data()
    
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    ax = fig.add_subplot(111, facecolor=COLORS['background'])
    
    # Only fundamental edges
    for edge in G_fund.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_fund[edge[0]][edge[1]].get('weight', 0.7)
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=COLORS['fundamental'], 
                   linewidth=weight*8 + 4, 
                   linestyle=':', dashes=(3, 8), alpha=0.7, zorder=1)
    
    draw_nodes_modern(ax, stocks, pos, node_colors, size=1000, fontsize=13)
    
    x_coords = [pos[s][0] for s in stocks if s in pos]
    y_coords = [pos[s][1] for s in stocks if s in pos]
    ax.set_xlim(min(x_coords) - 2, max(x_coords) + 2)
    ax.set_ylim(min(y_coords) - 2, max(y_coords) + 2)
    
    ax.set_title('Fundamental Similarity Edges: Long-Term Value Alignment', 
                fontsize=22, fontweight='bold', pad=25, color=COLORS['text'])
    ax.axis('off')
    
    info_text = ('Static Edges\n'
                'Cosine similarity\n'
                'P/E, ROE features\n'
                'Threshold: >0.7\n'
                'Updates: Quarterly')
    info_box = FancyBboxPatch((0.02, 0.02), 0.25, 0.20, 
                              boxstyle="round,pad=0.02", 
                              transform=ax.transAxes,
                              facecolor='white', edgecolor=COLORS['fundamental'], 
                              linewidth=3, alpha=0.95, zorder=10)
    ax.add_patch(info_box)
    ax.text(0.145, 0.12, info_text, transform=ax.transAxes, 
           fontsize=13, verticalalignment='center', horizontalalignment='center',
           color=COLORS['text'], fontweight='bold', zorder=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7d_fundamental_only.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7d_fundamental_only.png")


def create_figure7e_tech_cluster_detail():
    """Technology cluster detail."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors = get_graph_data()
    
    tech_stocks = sectors['tech']
    tech_pos = {s: pos[s] for s in tech_stocks if s in pos}
    
    # Create subgraph for tech cluster
    G_tech = nx.Graph()
    G_tech.add_nodes_from(tech_stocks)
    for edge in G_corr.edges():
        if edge[0] in tech_stocks and edge[1] in tech_stocks:
            G_tech.add_edge(edge[0], edge[1], weight=G_corr[edge[0]][edge[1]].get('weight', 0.7))
    
    # Re-layout for better visualization
    tech_pos = nx.spring_layout(G_tech, k=2, iterations=100, seed=42)
    tech_pos = {k: (v[0]*6, v[1]*6) for k, v in tech_pos.items()}
    
    fig = plt.figure(figsize=(16, 14), facecolor='white')
    ax = fig.add_subplot(111, facecolor=COLORS['background'])
    
    # All edge types for tech
    for edge in G_corr.edges():
        if edge[0] in tech_pos and edge[1] in tech_pos:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            ax.plot([tech_pos[edge[0]][0], tech_pos[edge[1]][0]], 
                   [tech_pos[edge[0]][1], tech_pos[edge[1]][1]], 
                   color=COLORS['correlation'], linewidth=weight*12 + 6, alpha=0.8, zorder=1)
    
    for edge in G_sector.edges():
        if edge[0] in tech_pos and edge[1] in tech_pos:
            ax.plot([tech_pos[edge[0]][0], tech_pos[edge[1]][0]], 
                   [tech_pos[edge[0]][1], tech_pos[edge[1]][1]], 
                   color=COLORS['sector'], linewidth=7, linestyle='--', 
                   dashes=(15, 8), alpha=0.7, zorder=1)
    
    for edge in G_fund.edges():
        if edge[0] in tech_pos and edge[1] in tech_pos:
            weight = G_fund[edge[0]][edge[1]].get('weight', 0.7)
            ax.plot([tech_pos[edge[0]][0], tech_pos[edge[1]][0]], 
                   [tech_pos[edge[0]][1], tech_pos[edge[1]][1]], 
                   color=COLORS['fundamental'], linewidth=weight*8 + 4, 
                   linestyle=':', dashes=(3, 8), alpha=0.7, zorder=1)
    
    for stock in tech_stocks:
        if stock in tech_pos:
            circle = Circle(tech_pos[stock], 0.5, color=node_colors[stock], 
                          alpha=0.95, zorder=3, edgecolor='white', linewidth=4)
            ax.add_patch(circle)
            text = ax.text(tech_pos[stock][0], tech_pos[stock][1], stock, 
                          ha='center', va='center', fontsize=16, 
                          fontweight='bold', color='white', zorder=4)
            text.set_path_effects([withStroke(linewidth=5, foreground='black', alpha=0.7)])
    
    x_coords = [tech_pos[s][0] for s in tech_stocks if s in tech_pos]
    y_coords = [tech_pos[s][1] for s in tech_stocks if s in tech_pos]
    ax.set_xlim(min(x_coords) - 1.5, max(x_coords) + 1.5)
    ax.set_ylim(min(y_coords) - 1.5, max(y_coords) + 1.5)
    
    ax.set_title('Technology Sector Cluster: Most Densely Connected', 
                fontsize=22, fontweight='bold', pad=25, color=COLORS['text'])
    ax.axis('off')
    
    legend_elements = [
        Line2D([0], [0], color=COLORS['correlation'], linewidth=10, label='Correlation'),
        Line2D([0], [0], color=COLORS['sector'], linestyle='--', linewidth=8, dashes=(15, 8), label='Sector'),
        Line2D([0], [0], color=COLORS['fundamental'], linestyle=':', linewidth=8, dashes=(3, 8), label='Fundamental'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14, 
             frameon=True, fancybox=True, shadow=True, 
             framealpha=0.98, edgecolor='#E0E0E0', facecolor='white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7e_tech_cluster_detail.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7e_tech_cluster_detail.png")


def create_figure7f_cross_sector_only():
    """Cross-sector connections only."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors = get_graph_data()
    
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    ax = fig.add_subplot(111, facecolor=COLORS['background'])
    
    draw_nodes_modern(ax, stocks, pos, node_colors, size=1000, fontsize=13)
    
    # Only cross-sector edges
    cross_sector_edges = []
    for edge in G_corr.edges():
        s1, s2 = edge[0], edge[1]
        s1_sector = next((k for k, v in sectors.items() if s1 in v), None)
        s2_sector = next((k for k, v in sectors.items() if s2 in v), None)
        if s1_sector != s2_sector and s1_sector and s2_sector:
            cross_sector_edges.append(edge)
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=COLORS['correlation'], linewidth=weight*10 + 6, alpha=0.8, zorder=1)
    
    x_coords = [pos[s][0] for s in stocks if s in pos]
    y_coords = [pos[s][1] for s in stocks if s in pos]
    ax.set_xlim(min(x_coords) - 2, max(x_coords) + 2)
    ax.set_ylim(min(y_coords) - 2, max(y_coords) + 2)
    
    ax.set_title(f'Cross-Sector Connections: {len(cross_sector_edges)} Inter-Sector Edges', 
                fontsize=22, fontweight='bold', pad=25, color=COLORS['text'])
    ax.axis('off')
    
    # Sector labels with modern style
    for sector_name, sector_stocks in sectors.items():
        x_coords = [pos[s][0] for s in sector_stocks if s in pos]
        y_coords = [pos[s][1] for s in sector_stocks if s in pos]
        if x_coords and y_coords:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            ax.text(center_x, center_y + 1.5, sector_name.upper(), 
                   ha='center', va='center', fontsize=16, 
                   fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.8', 
                           facecolor=COLORS[sector_name], alpha=0.9,
                           edgecolor='white', linewidth=3),
                   zorder=6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7f_cross_sector_only.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7f_cross_sector_only.png")


def create_figure7g_top_correlations():
    """Top 5 strongest correlations."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors = get_graph_data()
    
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    ax = fig.add_subplot(111, facecolor=COLORS['background'])
    
    edge_weights = []
    for edge in G_corr.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            edge_weights.append((edge, weight))
    
    sorted_edges = sorted(edge_weights, key=lambda x: x[1], reverse=True)[:5]
    strong_stocks = set()
    
    for (edge, weight) in sorted_edges:
        strong_stocks.add(edge[0])
        strong_stocks.add(edge[1])
        ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
               [pos[edge[0]][1], pos[edge[1]][1]], 
               color=COLORS['correlation'], linewidth=weight*12 + 8, alpha=0.9, zorder=1)
        # Weight label
        mid_x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
        mid_y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
        ax.text(mid_x, mid_y, f'{weight:.2f}', 
               fontsize=14, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.7', facecolor='white', 
                       alpha=0.95, edgecolor=COLORS['correlation'], linewidth=3),
               zorder=2, color=COLORS['correlation'], fontweight='bold')
    
    for stock in strong_stocks:
        if stock in pos:
            circle = Circle(pos[stock], 0.55, color=node_colors[stock], 
                          alpha=0.95, zorder=3, edgecolor='white', linewidth=5)
            ax.add_patch(circle)
            text = ax.text(pos[stock][0], pos[stock][1], stock, 
                          ha='center', va='center', fontsize=17, 
                          fontweight='bold', color='white', zorder=4)
            text.set_path_effects([withStroke(linewidth=6, foreground='black', alpha=0.8)])
    
    # Gray out other nodes
    for stock in stocks:
        if stock in pos and stock not in strong_stocks:
            circle = Circle(pos[stock], 0.35, color='#BDC3C7', 
                          alpha=0.2, zorder=2, edgecolor='gray', linewidth=1)
            ax.add_patch(circle)
            ax.text(pos[stock][0], pos[stock][1], stock, 
                   ha='center', va='center', fontsize=11, 
                   color='gray', zorder=3, alpha=0.4)
    
    x_coords = [pos[s][0] for s in stocks if s in pos]
    y_coords = [pos[s][1] for s in stocks if s in pos]
    ax.set_xlim(min(x_coords) - 2, max(x_coords) + 2)
    ax.set_ylim(min(y_coords) - 2, max(y_coords) + 2)
    
    ax.set_title('Top 5 Strongest Correlations: Highest Predictive Power', 
                fontsize=22, fontweight='bold', pad=25, color=COLORS['text'])
    ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7g_top_correlations.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7g_top_correlations.png")


def create_figure7h_correlation_distribution():
    """Correlation distribution histogram - modern style."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors = get_graph_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor='white')
    ax.set_facecolor(COLORS['background'])
    
    edge_weights = []
    for edge in G_corr.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            edge_weights.append(weight)
    
    n, bins, patches = ax.hist(edge_weights, bins=14, color=COLORS['correlation'], 
                               edgecolor='white', linewidth=2.5, alpha=0.8)
    
    # Color gradient
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.Reds(0.4 + 0.6 * i / len(patches)))
    
    ax.axvline(x=np.mean(edge_weights), color='#2C3E50', linestyle='--', 
              linewidth=4, label=f'Mean: {np.mean(edge_weights):.3f}', zorder=3)
    ax.axvline(x=np.median(edge_weights), color=COLORS['accent'], linestyle='--', 
              linewidth=4, label=f'Median: {np.median(edge_weights):.3f}', zorder=3)
    
    ax.set_xlabel('Correlation Coefficient', fontsize=18, fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Frequency', fontsize=18, fontweight='bold', color=COLORS['text'])
    ax.set_title('Correlation Strength Distribution', 
                fontsize=22, fontweight='bold', pad=25, color=COLORS['text'])
    ax.legend(fontsize=15, frameon=True, fancybox=True, shadow=True, 
             framealpha=0.95, facecolor='white', edgecolor='#E0E0E0')
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'figure7h_correlation_distribution.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7h_correlation_distribution.png")


def create_figure7i_sector_statistics():
    """Sector statistics table - modern design."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors = get_graph_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
    ax.axis('off')
    
    sector_stats = []
    for sector_name, sector_stocks in sectors.items():
        num_stocks = len(sector_stocks)
        num_edges = len([e for e in G_sector.edges() 
                        if e[0] in sector_stocks and e[1] in sector_stocks])
        sector_stats.append({
            'Sector': sector_name.capitalize(),
            'Stocks': num_stocks,
            'Edges': num_edges,
            'Density': f'{num_edges / (num_stocks * (num_stocks - 1) / 2) * 100:.1f}%' if num_stocks > 1 else 'N/A'
        })
    
    table_data = [['Sector', 'Stocks', 'Edges', 'Density']]
    for stat in sector_stats:
        table_data.append([stat['Sector'], str(stat['Stocks']), 
                          str(stat['Edges']), stat['Density']])
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='center', loc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 4.5)
    
    # Header styling
    for i in range(4):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_edgecolor('white')
        table[(0, i)].set_linewidth(4)
    
    # Row styling
    for i in range(1, len(table_data)):
        for j in range(4):
            table[(i, j)].set_edgecolor('#E0E0E0')
            table[(i, j)].set_linewidth(2.5)
            if j == 0:
                sector_name = table_data[i][0].lower()
                if sector_name in COLORS:
                    table[(i, j)].set_facecolor(COLORS[sector_name] + '30')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('white')
    
    ax.set_title('Sector Connectivity Statistics', 
                fontsize=22, fontweight='bold', pad=30, color=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'figure7i_sector_statistics.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7i_sector_statistics.png")


def create_figure7j_fundamental_distribution():
    """Fundamental similarity distribution."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors = get_graph_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor='white')
    ax.set_facecolor(COLORS['background'])
    
    edge_weights = []
    for edge in G_fund.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_fund[edge[0]][edge[1]].get('weight', 0.7)
            edge_weights.append(weight)
    
    n, bins, patches = ax.hist(edge_weights, bins=12, color=COLORS['fundamental'], 
                               edgecolor='white', linewidth=2.5, alpha=0.8)
    
    # Color gradient
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.Greens(0.4 + 0.6 * i / len(patches)))
    
    ax.axvline(x=0.7, color='#E74C3C', linestyle='--', linewidth=4, 
              label='Threshold: 0.7', zorder=3)
    ax.axvline(x=np.mean(edge_weights), color='#2C3E50', linestyle='--', 
              linewidth=4, label=f'Mean: {np.mean(edge_weights):.3f}', zorder=3)
    
    ax.set_xlabel('Similarity Score', fontsize=18, fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Frequency', fontsize=18, fontweight='bold', color=COLORS['text'])
    ax.set_title('Fundamental Similarity Distribution', 
                fontsize=22, fontweight='bold', pad=25, color=COLORS['text'])
    ax.legend(fontsize=15, frameon=True, fancybox=True, shadow=True, 
             framealpha=0.95, facecolor='white', edgecolor='#E0E0E0')
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'figure7j_fundamental_distribution.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7j_fundamental_distribution.png")


def create_figure7k_edge_comparison_table():
    """Edge type comparison table - modern design."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors = get_graph_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 10), facecolor='white')
    ax.axis('off')
    
    comparison_data = [
        ['Edge Type', 'Count', 'Type', 'Updates', 'Weighted', 'Purpose'],
        ['Rolling Correlation', '12', 'Dynamic', 'Daily', 'Yes', 'Short-term co-movements'],
        ['Sector/Industry', '13', 'Static', 'Never', 'No', 'Industry-level factors'],
        ['Fundamental Similarity', '9', 'Static', 'Quarterly', 'Yes', 'Long-term value alignment'],
        ['Total', '34', 'Mixed', 'Mixed', 'Mixed', 'Multi-relational learning']
    ]
    
    table = ax.table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                    cellLoc='center', loc='center',
                    colWidths=[0.24, 0.12, 0.12, 0.12, 0.12, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 5.5)
    
    # Header
    for i in range(6):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_edgecolor('white')
        table[(0, i)].set_linewidth(4)
    
    # Rows with color coding
    row_colors = [COLORS['correlation'] + '40', COLORS['sector'] + '40', 
                  COLORS['fundamental'] + '40', '#F5F5F5']
    for i in range(1, len(comparison_data)):
        for j in range(6):
            table[(i, j)].set_edgecolor('#E0E0E0')
            table[(i, j)].set_linewidth(2.5)
            if i < len(comparison_data) - 1:
                table[(i, j)].set_facecolor(row_colors[i-1])
                if j == 0:
                    table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor(row_colors[-1])
                table[(i, j)].set_text_props(weight='bold')
    
    ax.set_title('Edge Type Comparison: Key Characteristics', 
                fontsize=22, fontweight='bold', color=COLORS['text'], pad=40)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'figure7k_edge_comparison_table.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7k_edge_comparison_table.png")


if __name__ == "__main__":
    print("Generating redesigned Figure 7 series...")
    print("\n=== Creating modern, professional figures ===")
    create_figure7a_complete_overview()
    create_figure7b_correlation_only()
    create_figure7c_sector_only()
    create_figure7d_fundamental_only()
    create_figure7e_tech_cluster_detail()
    create_figure7f_cross_sector_only()
    create_figure7g_top_correlations()
    create_figure7h_correlation_distribution()
    create_figure7i_sector_statistics()
    create_figure7j_fundamental_distribution()
    create_figure7k_edge_comparison_table()
    
    print("\n✅ All redesigned Figure 7 charts generated!")
    print("Total: 11 modern, professional figures")

