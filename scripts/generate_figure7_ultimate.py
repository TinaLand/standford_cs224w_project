"""
Ultimate Figure 7 series - Each figure is completely unique with distinct visual styles
No repetition, maximum visual appeal, professional design
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import seaborn as sns

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGS_DIR = PROJECT_ROOT / "figures"
FIGS_DIR.mkdir(exist_ok=True)

# Professional, modern color palette
PALETTE = {
    'tech': '#FF6B9D',      # Pink
    'finance': '#4ECDC4',   # Teal
    'healthcare': '#95E1D3', # Mint
    'consumer': '#F38181',  # Coral
    'energy': '#AA96DA',    # Lavender
    'correlation': '#FF6B9D',
    'sector': '#4ECDC4',
    'fundamental': '#95E1D3',
    'bg': '#FFFFFF',
    'bg_light': '#F8F9FA',
    'text': '#1A1A1A',
    'text_light': '#6C757D',
    'accent': '#FFD93D',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545'
}

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']
sns.set_style("whitegrid", {'axes.grid': False})

def get_graph_data():
    """Get graph data."""
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'BAC', 'JNJ', 'PFE', 'WMT', 
              'HD', 'MCD', 'XOM', 'CVX']
    
    sectors = {
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'finance': ['JPM', 'BAC'],
        'healthcare': ['JNJ', 'PFE'],
        'consumer': ['WMT', 'HD', 'MCD'],
        'energy': ['XOM', 'CVX']
    }
    
    G_corr = nx.Graph()
    G_corr.add_nodes_from(stocks)
    corr_edges = [
        ('AAPL', 'MSFT', 0.85), ('AAPL', 'GOOGL', 0.78), ('MSFT', 'GOOGL', 0.82),
        ('AMZN', 'META', 0.75), ('AMZN', 'GOOGL', 0.70), ('META', 'GOOGL', 0.72),
        ('JPM', 'BAC', 0.88), ('JNJ', 'PFE', 0.80), ('WMT', 'HD', 0.65),
        ('XOM', 'CVX', 0.90), ('AAPL', 'AMZN', 0.68), ('MSFT', 'META', 0.65)
    ]
    G_corr.add_weighted_edges_from(corr_edges)
    
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
    
    G_fund = nx.Graph()
    G_fund.add_nodes_from(stocks)
    fund_edges = [
        ('AAPL', 'MSFT', 0.82), ('GOOGL', 'META', 0.75), ('AMZN', 'META', 0.70),
        ('JPM', 'BAC', 0.85), ('JNJ', 'PFE', 0.78), ('XOM', 'CVX', 0.88),
        ('WMT', 'HD', 0.72), ('HD', 'MCD', 0.68), ('AAPL', 'AMZN', 0.65)
    ]
    G_fund.add_weighted_edges_from(fund_edges)
    
    # Use different layouts for different figures
    pos_circular = nx.circular_layout(G_corr)
    pos_spring = nx.spring_layout(G_corr, k=2.5, iterations=100, seed=42)
    pos_kamada = nx.kamada_kawai_layout(G_corr)
    
    # Scale positions
    pos_circular = {k: (v[0]*10, v[1]*10) for k, v in pos_circular.items()}
    pos_spring = {k: (v[0]*8, v[1]*8) for k, v in pos_spring.items()}
    pos_kamada = {k: (v[0]*9, v[1]*9) for k, v in pos_kamada.items()}
    
    node_colors = {}
    for sector_name, sector_stocks in sectors.items():
        for stock in sector_stocks:
            node_colors[stock] = PALETTE[sector_name]
    
    return stocks, sectors, G_corr, G_sector, G_fund, pos_circular, pos_spring, pos_kamada, node_colors


def draw_node_modern(ax, x, y, label, color, size=0.4, fontsize=12, style='default'):
    """Draw a modern node with different styles."""
    if style == 'glow':
        # Glow effect
        for r, alpha in [(size+0.2, 0.3), (size+0.1, 0.5), (size, 0.8)]:
            circle = Circle((x, y), r, color=color, alpha=alpha, zorder=1)
            ax.add_patch(circle)
        circle = Circle((x, y), size, color=color, alpha=0.95, zorder=3, 
                       edgecolor='white', linewidth=3)
        ax.add_patch(circle)
    elif style == 'shadow':
        # Shadow effect
        shadow = Circle((x+0.1, y-0.1), size, color='black', alpha=0.2, zorder=1)
        ax.add_patch(shadow)
        circle = Circle((x, y), size, color=color, alpha=0.95, zorder=3, 
                       edgecolor='white', linewidth=3)
        ax.add_patch(circle)
    else:  # default
        circle = Circle((x, y), size, color=color, alpha=0.9, zorder=3, 
                       edgecolor='white', linewidth=3)
        ax.add_patch(circle)
    
    text = ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, 
                  fontweight='bold', color='white', zorder=4)
    text.set_path_effects([withStroke(linewidth=4, foreground='black', alpha=0.6)])


# ============================================================================
# UNIQUE FIGURE 7 SERIES - Each with distinct visual style
# ============================================================================

def create_figure7a_complete_overview():
    """7a: Complete overview - Force-directed layout with all edges."""
    stocks, sectors, G_corr, G_sector, G_fund, pos_c, pos_s, pos_k, node_colors = get_graph_data()
    pos = pos_s  # Use spring layout
    
    fig = plt.figure(figsize=(22, 18), facecolor='white')
    ax = fig.add_subplot(111, facecolor=PALETTE['bg_light'])
    
    # Draw all edges with distinct styles
    for edge in G_corr.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=PALETTE['correlation'], linewidth=weight*10 + 5, 
                   alpha=0.65, zorder=1, solid_capstyle='round')
    
    for edge in G_sector.edges():
        if edge[0] in pos and edge[1] in pos and edge not in G_corr.edges():
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=PALETTE['sector'], linewidth=6, linestyle='--', 
                   dashes=(20, 10), alpha=0.6, zorder=1)
    
    for edge in G_fund.edges():
        if edge[0] in pos and edge[1] in pos and edge not in G_corr.edges() and edge not in G_sector.edges():
            weight = G_fund[edge[0]][edge[1]].get('weight', 0.7)
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=PALETTE['fundamental'], linewidth=weight*8 + 4, 
                   linestyle=':', dashes=(4, 10), alpha=0.6, zorder=1)
    
    for stock in stocks:
        if stock in pos:
            draw_node_modern(ax, pos[stock][0], pos[stock][1], stock, 
                           node_colors[stock], size=0.5, fontsize=14, style='glow')
    
    x_coords = [pos[s][0] for s in stocks if s in pos]
    y_coords = [pos[s][1] for s in stocks if s in pos]
    ax.set_xlim(min(x_coords) - 3, max(x_coords) + 3)
    ax.set_ylim(min(y_coords) - 3, max(y_coords) + 3)
    
    ax.set_title('Heterogeneous Stock Graph: Multi-Relational Structure', 
                fontsize=26, fontweight='bold', pad=40, color=PALETTE['text'])
    ax.axis('off')
    
    # Modern legend in top-right
    legend_elements = [
        Line2D([0], [0], color=PALETTE['correlation'], linewidth=12, label='Rolling Correlation'),
        Line2D([0], [0], color=PALETTE['sector'], linestyle='--', linewidth=10, 
              dashes=(20, 10), label='Sector/Industry'),
        Line2D([0], [0], color=PALETTE['fundamental'], linestyle=':', linewidth=10, 
              dashes=(4, 10), label='Fundamental Similarity'),
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=17, 
                      frameon=True, fancybox=True, shadow=True, 
                      framealpha=0.98, edgecolor='#D0D0D0', facecolor='white',
                      bbox_to_anchor=(0.98, 0.98), ncol=1)
    legend.get_frame().set_linewidth(2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7a_complete_overview.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✅ Created: figure7a_complete_overview.png")


def create_figure7b_correlation_only():
    """7b: Correlation edges ONLY - Circular layout, heatmap style."""
    stocks, sectors, G_corr, G_sector, G_fund, pos_c, pos_s, pos_k, node_colors = get_graph_data()
    pos = pos_c  # Use circular layout for this one
    
    fig = plt.figure(figsize=(20, 20), facecolor='white')
    ax = fig.add_subplot(111, facecolor=PALETTE['bg'])
    
    # Only correlation edges with gradient colors
    edge_weights = []
    for edge in G_corr.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            edge_weights.append(weight)
            # Color based on weight
            color_intensity = weight
            color = plt.cm.Reds(0.3 + 0.7 * color_intensity)
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=color, linewidth=weight*12 + 6, 
                   alpha=0.75, zorder=1, solid_capstyle='round')
    
    for stock in stocks:
        if stock in pos:
            draw_node_modern(ax, pos[stock][0], pos[stock][1], stock, 
                           node_colors[stock], size=0.6, fontsize=15, style='shadow')
    
    x_coords = [pos[s][0] for s in stocks if s in pos]
    y_coords = [pos[s][1] for s in stocks if s in pos]
    # Add more padding to ensure nothing is cut off
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    padding = max(x_range, y_range) * 0.15  # 15% padding
    ax.set_xlim(min(x_coords) - padding, max(x_coords) + padding)
    ax.set_ylim(min(y_coords) - padding, max(y_coords) + padding)
    
    ax.set_title('Rolling Correlation Edges: Dynamic Price Co-Movements', 
                fontsize=24, fontweight='bold', pad=30, color=PALETTE['text'])
    ax.axis('off')
    
    # Info card - move to upper right corner to avoid blocking graph
    info_box = FancyBboxPatch((0.75, 0.85), 0.22, 0.13, 
                              boxstyle="round,pad=0.02", 
                              transform=ax.transAxes,
                              facecolor='white', edgecolor=PALETTE['correlation'], 
                              linewidth=3, alpha=0.98, zorder=10)
    ax.add_patch(info_box)
    info_text = ('Dynamic Edges\n'
                '30-day rolling window\n'
                'Top-K=10 per stock\n'
                'Updates: Daily')
    ax.text(0.86, 0.915, info_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='center', horizontalalignment='center',
           color=PALETTE['text'], fontweight='bold', zorder=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7b_correlation_only.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✅ Created: figure7b_correlation_only.png")


def create_figure7c_sector_only():
    """7c: Sector edges ONLY - Kamada-Kawai layout, cluster visualization."""
    stocks, sectors, G_corr, G_sector, G_fund, pos_c, pos_s, pos_k, node_colors = get_graph_data()
    pos = pos_k  # Use Kamada-Kawai layout
    
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    ax = fig.add_subplot(111, facecolor=PALETTE['bg_light'])
    
    # Draw sector clusters with background circles
    for sector_name, sector_stocks in sectors.items():
        x_coords = [pos[s][0] for s in sector_stocks if s in pos]
        y_coords = [pos[s][1] for s in sector_stocks if s in pos]
        if x_coords and y_coords:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            radius = max([np.sqrt((x-center_x)**2 + (y-center_y)**2) 
                         for x, y in zip(x_coords, y_coords)]) + 1.5
            circle = Circle((center_x, center_y), radius, 
                          color=PALETTE[sector_name], alpha=0.15, zorder=0)
            ax.add_patch(circle)
    
    # Only sector edges
    for edge in G_sector.edges():
        if edge[0] in pos and edge[1] in pos:
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=PALETTE['sector'], linewidth=7, linestyle='--', 
                   dashes=(20, 10), alpha=0.75, zorder=2)
    
    for stock in stocks:
        if stock in pos:
            draw_node_modern(ax, pos[stock][0], pos[stock][1], stock, 
                           node_colors[stock], size=0.5, fontsize=14, style='default')
    
    x_coords = [pos[s][0] for s in stocks if s in pos]
    y_coords = [pos[s][1] for s in stocks if s in pos]
    ax.set_xlim(min(x_coords) - 2, max(x_coords) + 2)
    ax.set_ylim(min(y_coords) - 2, max(y_coords) + 2)
    
    ax.set_title('Sector/Industry Edges: Static Domain Knowledge', 
                fontsize=24, fontweight='bold', pad=30, color=PALETTE['text'])
    ax.axis('off')
    
    # Info card (smaller)
    info_box = FancyBboxPatch((0.02, 0.02), 0.20, 0.13, 
                              boxstyle="round,pad=0.02", 
                              transform=ax.transAxes,
                              facecolor='white', edgecolor=PALETTE['sector'], 
                              linewidth=3, alpha=0.98, zorder=10)
    ax.add_patch(info_box)
    info_text = ('Static Edges\n'
                'Industry classification\n'
                'Binary (same sector=1)\n'
                'Never updated')
    ax.text(0.12, 0.085, info_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='center', horizontalalignment='center',
           color=PALETTE['text'], fontweight='bold', zorder=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7c_sector_only.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✅ Created: figure7c_sector_only.png")


def create_figure7d_fundamental_only():
    """7d: Fundamental edges ONLY - Spring layout with weight visualization."""
    stocks, sectors, G_corr, G_sector, G_fund, pos_c, pos_s, pos_k, node_colors = get_graph_data()
    pos = pos_s  # Use spring layout
    
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    ax = fig.add_subplot(111, facecolor=PALETTE['bg'])
    
    # Only fundamental edges with weight labels
    for edge in G_fund.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_fund[edge[0]][edge[1]].get('weight', 0.7)
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=PALETTE['fundamental'], linewidth=weight*10 + 5, 
                   linestyle=':', dashes=(4, 10), alpha=0.7, zorder=1)
            # Add weight label
            mid_x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
            mid_y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
            ax.text(mid_x, mid_y, f'{weight:.2f}', 
                   fontsize=11, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                           alpha=0.9, edgecolor=PALETTE['fundamental'], linewidth=2),
                   zorder=2, color=PALETTE['fundamental'], fontweight='bold')
    
    for stock in stocks:
        if stock in pos:
            draw_node_modern(ax, pos[stock][0], pos[stock][1], stock, 
                           node_colors[stock], size=0.5, fontsize=14, style='glow')
    
    x_coords = [pos[s][0] for s in stocks if s in pos]
    y_coords = [pos[s][1] for s in stocks if s in pos]
    ax.set_xlim(min(x_coords) - 2, max(x_coords) + 2)
    ax.set_ylim(min(y_coords) - 2, max(y_coords) + 2)
    
    ax.set_title('Fundamental Similarity Edges: Long-Term Value Alignment', 
                fontsize=24, fontweight='bold', pad=30, color=PALETTE['text'])
    ax.axis('off')
    
    # Info card (smaller)
    info_box = FancyBboxPatch((0.02, 0.02), 0.20, 0.16, 
                              boxstyle="round,pad=0.02", 
                              transform=ax.transAxes,
                              facecolor='white', edgecolor=PALETTE['fundamental'], 
                              linewidth=3, alpha=0.98, zorder=10)
    ax.add_patch(info_box)
    info_text = ('Static Edges\n'
                'Cosine similarity\n'
                'P/E, ROE features\n'
                'Threshold: >0.7\n'
                'Updates: Quarterly')
    ax.text(0.12, 0.10, info_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='center', horizontalalignment='center',
           color=PALETTE['text'], fontweight='bold', zorder=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7d_fundamental_only.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✅ Created: figure7d_fundamental_only.png")


def create_figure7e_tech_cluster_detail():
    """7e: Tech cluster ONLY - Concentrated circular layout."""
    stocks, sectors, G_corr, G_sector, G_fund, pos_c, pos_s, pos_k, node_colors = get_graph_data()
    
    tech_stocks = sectors['tech']
    G_tech = nx.Graph()
    G_tech.add_nodes_from(tech_stocks)
    for edge in G_corr.edges():
        if edge[0] in tech_stocks and edge[1] in tech_stocks:
            G_tech.add_edge(edge[0], edge[1], weight=G_corr[edge[0]][edge[1]].get('weight', 0.7))
    
    # Concentrated layout for tech cluster
    pos_tech = nx.spring_layout(G_tech, k=1.5, iterations=100, seed=42)
    pos_tech = {k: (v[0]*5, v[1]*5) for k, v in pos_tech.items()}
    
    fig = plt.figure(figsize=(18, 18), facecolor='white')
    ax = fig.add_subplot(111, facecolor=PALETTE['bg_light'])
    
    # All edge types for tech
    for edge in G_corr.edges():
        if edge[0] in pos_tech and edge[1] in pos_tech:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            ax.plot([pos_tech[edge[0]][0], pos_tech[edge[1]][0]], 
                   [pos_tech[edge[0]][1], pos_tech[edge[1]][1]], 
                   color=PALETTE['correlation'], linewidth=weight*15 + 8, alpha=0.8, zorder=1)
    
    for edge in G_sector.edges():
        if edge[0] in pos_tech and edge[1] in pos_tech:
            ax.plot([pos_tech[edge[0]][0], pos_tech[edge[1]][0]], 
                   [pos_tech[edge[0]][1], pos_tech[edge[1]][1]], 
                   color=PALETTE['sector'], linewidth=8, linestyle='--', 
                   dashes=(20, 10), alpha=0.75, zorder=1)
    
    for edge in G_fund.edges():
        if edge[0] in pos_tech and edge[1] in pos_tech:
            weight = G_fund[edge[0]][edge[1]].get('weight', 0.7)
            ax.plot([pos_tech[edge[0]][0], pos_tech[edge[1]][0]], 
                   [pos_tech[edge[0]][1], pos_tech[edge[1]][1]], 
                   color=PALETTE['fundamental'], linewidth=weight*10 + 5, 
                   linestyle=':', dashes=(4, 10), alpha=0.75, zorder=1)
    
    for stock in tech_stocks:
        if stock in pos_tech:
            draw_node_modern(ax, pos_tech[stock][0], pos_tech[stock][1], stock, 
                           node_colors[stock], size=0.7, fontsize=18, style='glow')
    
    x_coords = [pos_tech[s][0] for s in tech_stocks if s in pos_tech]
    y_coords = [pos_tech[s][1] for s in tech_stocks if s in pos_tech]
    ax.set_xlim(min(x_coords) - 2, max(x_coords) + 2)
    ax.set_ylim(min(y_coords) - 2, max(y_coords) + 2)
    
    ax.set_title('Technology Sector Cluster: Most Densely Connected', 
                fontsize=24, fontweight='bold', pad=30, color=PALETTE['text'])
    ax.axis('off')
    
    legend_elements = [
        Line2D([0], [0], color=PALETTE['correlation'], linewidth=12, label='Correlation'),
        Line2D([0], [0], color=PALETTE['sector'], linestyle='--', linewidth=10, 
              dashes=(20, 10), label='Sector'),
        Line2D([0], [0], color=PALETTE['fundamental'], linestyle=':', linewidth=10, 
              dashes=(4, 10), label='Fundamental'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=15, 
             frameon=True, fancybox=True, shadow=True, 
             framealpha=0.98, edgecolor='#D0D0D0', facecolor='white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7e_tech_cluster_detail.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✅ Created: figure7e_tech_cluster_detail.png")


def create_figure7f_cross_sector_only():
    """7f: Cross-sector connections ONLY - Highlighted bridges."""
    stocks, sectors, G_corr, G_sector, G_fund, pos_c, pos_s, pos_k, node_colors = get_graph_data()
    pos = pos_k  # Use Kamada-Kawai for better separation
    
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    ax = fig.add_subplot(111, facecolor=PALETTE['bg'])
    
    # Draw all nodes first (grayed out)
    for stock in stocks:
        if stock in pos:
            draw_node_modern(ax, pos[stock][0], pos[stock][1], stock, 
                           '#BDC3C7', size=0.4, fontsize=12, style='default')
    
    # Highlight cross-sector edges
    cross_sector_edges = []
    for edge in G_corr.edges():
        s1, s2 = edge[0], edge[1]
        s1_sector = next((k for k, v in sectors.items() if s1 in v), None)
        s2_sector = next((k for k, v in sectors.items() if s2 in v), None)
        if s1_sector != s2_sector and s1_sector and s2_sector:
            cross_sector_edges.append(edge)
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            # Highlighted edge
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                   [pos[edge[0]][1], pos[edge[1]][1]], 
                   color=PALETTE['accent'], linewidth=weight*12 + 8, alpha=0.9, zorder=2)
            # Redraw nodes in color
            for s in [edge[0], edge[1]]:
                if s in pos:
                    draw_node_modern(ax, pos[s][0], pos[s][1], s, 
                                   node_colors[s], size=0.5, fontsize=14, style='glow')
    
    x_coords = [pos[s][0] for s in stocks if s in pos]
    y_coords = [pos[s][1] for s in stocks if s in pos]
    ax.set_xlim(min(x_coords) - 2, max(x_coords) + 2)
    ax.set_ylim(min(y_coords) - 2, max(y_coords) + 2)
    
    ax.set_title(f'Cross-Sector Connections: {len(cross_sector_edges)} Inter-Sector Edges', 
                fontsize=24, fontweight='bold', pad=30, color=PALETTE['text'])
    ax.axis('off')
    
    # Sector labels
    for sector_name, sector_stocks in sectors.items():
        x_coords = [pos[s][0] for s in sector_stocks if s in pos]
        y_coords = [pos[s][1] for s in sector_stocks if s in pos]
        if x_coords and y_coords:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            ax.text(center_x, center_y + 2, sector_name.upper(), 
                   ha='center', va='center', fontsize=17, 
                   fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=1.0', 
                           facecolor=PALETTE[sector_name], alpha=0.95,
                           edgecolor='white', linewidth=4),
                   zorder=6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7f_cross_sector_only.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✅ Created: figure7f_cross_sector_only.png")


def create_figure7g_top_correlations():
    """7g: Top 5 correlations - Network diagram style."""
    stocks, sectors, G_corr, G_sector, G_fund, pos_c, pos_s, pos_k, node_colors = get_graph_data()
    pos = pos_s  # Use spring layout
    
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    ax = fig.add_subplot(111, facecolor=PALETTE['bg_light'])
    
    edge_weights = []
    for edge in G_corr.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            edge_weights.append((edge, weight))
    
    sorted_edges = sorted(edge_weights, key=lambda x: x[1], reverse=True)[:5]
    strong_stocks = set()
    
    # Draw top edges
    for (edge, weight) in sorted_edges:
        strong_stocks.add(edge[0])
        strong_stocks.add(edge[1])
        ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
               [pos[edge[0]][1], pos[edge[1]][1]], 
               color=PALETTE['correlation'], linewidth=weight*15 + 10, alpha=0.9, zorder=1)
        # Weight label with arrow
        mid_x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2
        mid_y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2
        ax.text(mid_x, mid_y, f'{weight:.2f}', 
               fontsize=15, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                       alpha=0.98, edgecolor=PALETTE['correlation'], linewidth=4),
               zorder=2, color=PALETTE['correlation'], fontweight='bold')
    
    # Highlight strong nodes
    for stock in strong_stocks:
        if stock in pos:
            draw_node_modern(ax, pos[stock][0], pos[stock][1], stock, 
                           node_colors[stock], size=0.7, fontsize=17, style='glow')
    
    # Gray out other nodes
    for stock in stocks:
        if stock in pos and stock not in strong_stocks:
            circle = Circle(pos[stock], 0.3, color='#E0E0E0', 
                          alpha=0.3, zorder=0, edgecolor='gray', linewidth=1)
            ax.add_patch(circle)
            ax.text(pos[stock][0], pos[stock][1], stock, 
                   ha='center', va='center', fontsize=10, 
                   color='gray', zorder=1, alpha=0.5)
    
    x_coords = [pos[s][0] for s in stocks if s in pos]
    y_coords = [pos[s][1] for s in stocks if s in pos]
    ax.set_xlim(min(x_coords) - 2, max(x_coords) + 2)
    ax.set_ylim(min(y_coords) - 2, max(y_coords) + 2)
    
    ax.set_title('Top 5 Strongest Correlations: Highest Predictive Power', 
                fontsize=24, fontweight='bold', pad=30, color=PALETTE['text'])
    ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS_DIR / 'figure7g_top_correlations.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.3)
    plt.close()
    print(f"✅ Created: figure7g_top_correlations.png")


def create_figure7h_correlation_distribution():
    """7h: Correlation distribution - Modern histogram with gradient."""
    stocks, sectors, G_corr, G_sector, G_fund, pos_c, pos_s, pos_k, node_colors = get_graph_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12), facecolor='white')
    ax.set_facecolor(PALETTE['bg'])
    
    edge_weights = []
    for edge in G_corr.edges():
        if edge[0] in pos_s and edge[1] in pos_s:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            edge_weights.append(weight)
    
    # Modern histogram with gradient
    n, bins, patches = ax.hist(edge_weights, bins=16, color=PALETTE['correlation'], 
                               edgecolor='white', linewidth=3, alpha=0.85)
    
    # Gradient fill
    for i, patch in enumerate(patches):
        intensity = i / len(patches)
        patch.set_facecolor(plt.cm.Reds(0.4 + 0.6 * intensity))
    
    mean_val = np.mean(edge_weights)
    median_val = np.median(edge_weights)
    
    ax.axvline(x=mean_val, color=PALETTE['text'], linestyle='--', 
              linewidth=5, label=f'Mean: {mean_val:.3f}', zorder=3)
    ax.axvline(x=median_val, color=PALETTE['accent'], linestyle='--', 
              linewidth=5, label=f'Median: {median_val:.3f}', zorder=3)
    
    ax.set_xlabel('Correlation Coefficient', fontsize=20, fontweight='bold', 
                 color=PALETTE['text'], labelpad=15)
    ax.set_ylabel('Frequency', fontsize=20, fontweight='bold', 
                 color=PALETTE['text'], labelpad=15)
    ax.set_title('Correlation Strength Distribution', 
                fontsize=24, fontweight='bold', pad=30, color=PALETTE['text'])
    ax.legend(fontsize=16, frameon=True, fancybox=True, shadow=True, 
             framealpha=0.98, facecolor='white', edgecolor='#D0D0D0', loc='upper right')
    ax.grid(True, alpha=0.15, axis='y', linestyle='--', linewidth=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')
    ax.tick_params(colors=PALETTE['text_light'], labelsize=14)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'figure7h_correlation_distribution.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7h_correlation_distribution.png")


def create_figure7i_sector_statistics():
    """7i: Sector statistics - Modern table design."""
    stocks, sectors, G_corr, G_sector, G_fund, pos_c, pos_s, pos_k, node_colors = get_graph_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor='white')
    ax.axis('off')
    
    sector_stats = []
    for sector_name, sector_stocks in sectors.items():
        num_stocks = len(sector_stocks)
        num_edges = len([e for e in G_sector.edges() 
                        if e[0] in sector_stocks and e[1] in sector_stocks])
        density = num_edges / (num_stocks * (num_stocks - 1) / 2) * 100 if num_stocks > 1 else 0
        sector_stats.append({
            'Sector': sector_name.capitalize(),
            'Stocks': num_stocks,
            'Edges': num_edges,
            'Density': f'{density:.1f}%' if num_stocks > 1 else 'N/A'
        })
    
    table_data = [['Sector', 'Stocks', 'Edges', 'Density']]
    for stat in sector_stats:
        table_data.append([stat['Sector'], str(stat['Stocks']), 
                          str(stat['Edges']), stat['Density']])
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='center', loc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(17)
    table.scale(1, 5.0)
    
    # Header
    for i in range(4):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_edgecolor('white')
        table[(0, i)].set_linewidth(5)
    
    # Rows with sector colors
    for i in range(1, len(table_data)):
        for j in range(4):
            table[(i, j)].set_edgecolor('#E0E0E0')
            table[(i, j)].set_linewidth(3)
            if j == 0:
                sector_name = table_data[i][0].lower()
                if sector_name in PALETTE:
                    table[(i, j)].set_facecolor(PALETTE[sector_name] + '40')
                table[(i, j)].set_text_props(weight='bold', size=18)
            else:
                table[(i, j)].set_facecolor('white')
                table[(i, j)].set_text_props(size=17)
    
    ax.set_title('Sector Connectivity Statistics', 
                fontsize=24, fontweight='bold', pad=40, color=PALETTE['text'])
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'figure7i_sector_statistics.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7i_sector_statistics.png")


def create_figure7j_fundamental_distribution():
    """7j: Fundamental distribution - Different style from correlation."""
    stocks, sectors, G_corr, G_sector, G_fund, pos_c, pos_s, pos_k, node_colors = get_graph_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12), facecolor='white')
    ax.set_facecolor(PALETTE['bg'])
    
    edge_weights = []
    for edge in G_fund.edges():
        if edge[0] in pos_s and edge[1] in pos_s:
            weight = G_fund[edge[0]][edge[1]].get('weight', 0.7)
            edge_weights.append(weight)
    
    # Different style - bar chart with gradient
    n, bins, patches = ax.hist(edge_weights, bins=14, color=PALETTE['fundamental'], 
                               edgecolor='white', linewidth=3, alpha=0.85)
    
    # Green gradient
    for i, patch in enumerate(patches):
        intensity = i / len(patches)
        patch.set_facecolor(plt.cm.Greens(0.4 + 0.6 * intensity))
    
    threshold = 0.7
    mean_val = np.mean(edge_weights)
    
    ax.axvline(x=threshold, color=PALETTE['danger'], linestyle='--', 
              linewidth=5, label='Threshold: 0.7', zorder=3)
    ax.axvline(x=mean_val, color=PALETTE['text'], linestyle='--', 
              linewidth=5, label=f'Mean: {mean_val:.3f}', zorder=3)
    
    ax.set_xlabel('Similarity Score', fontsize=20, fontweight='bold', 
                 color=PALETTE['text'], labelpad=15)
    ax.set_ylabel('Frequency', fontsize=20, fontweight='bold', 
                 color=PALETTE['text'], labelpad=15)
    ax.set_title('Fundamental Similarity Distribution', 
                fontsize=24, fontweight='bold', pad=30, color=PALETTE['text'])
    ax.legend(fontsize=16, frameon=True, fancybox=True, shadow=True, 
             framealpha=0.98, facecolor='white', edgecolor='#D0D0D0', loc='upper right')
    ax.grid(True, alpha=0.15, axis='y', linestyle='--', linewidth=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')
    ax.tick_params(colors=PALETTE['text_light'], labelsize=14)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'figure7j_fundamental_distribution.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7j_fundamental_distribution.png")


def create_figure7k_edge_comparison_table():
    """7k: Edge comparison table - Comprehensive comparison."""
    stocks, sectors, G_corr, G_sector, G_fund, pos_c, pos_s, pos_k, node_colors = get_graph_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), facecolor='white')
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
    table.set_fontsize(16)
    table.scale(1, 6.0)
    
    # Header
    for i in range(6):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white', size=18)
        table[(0, i)].set_edgecolor('white')
        table[(0, i)].set_linewidth(5)
    
    # Color-coded rows
    row_colors = [PALETTE['correlation'] + '50', PALETTE['sector'] + '50', 
                  PALETTE['fundamental'] + '50', '#F5F5F5']
    for i in range(1, len(comparison_data)):
        for j in range(6):
            table[(i, j)].set_edgecolor('#E0E0E0')
            table[(i, j)].set_linewidth(3)
            if i < len(comparison_data) - 1:
                table[(i, j)].set_facecolor(row_colors[i-1])
                if j == 0:
                    table[(i, j)].set_text_props(weight='bold', size=17)
                else:
                    table[(i, j)].set_text_props(size=16)
            else:
                table[(i, j)].set_facecolor(row_colors[-1])
                table[(i, j)].set_text_props(weight='bold', size=17)
    
    ax.set_title('Edge Type Comparison: Key Characteristics', 
                fontsize=24, fontweight='bold', color=PALETTE['text'], pad=50)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'figure7k_edge_comparison_table.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7k_edge_comparison_table.png")


if __name__ == "__main__":
    print("Generating ultimate Figure 7 series...")
    print("\n=== Creating unique, beautiful figures ===")
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
    
    print("\n✅ All ultimate Figure 7 charts generated!")
    print("Total: 11 completely unique, professional figures")

