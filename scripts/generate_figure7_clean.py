"""
Clean, simplified Figure 7 series - minimal text, no overlap, all elements fully visible
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
import networkx as nx

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGS_DIR = PROJECT_ROOT / "figures"
FIGS_DIR.mkdir(exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def _get_clean_graph_data():
    """Get graph data with better spacing."""
    colors = {
        'tech': '#E74C3C',
        'finance': '#3498DB',
        'healthcare': '#2ECC71',
        'consumer': '#F39C12',
        'energy': '#9B59B6',
    }
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'BAC', 'JNJ', 'PFE', 'WMT', 
              'HD', 'MCD', 'XOM', 'CVX']
    
    sectors = {
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'finance': ['JPM', 'BAC'],
        'healthcare': ['JNJ', 'PFE'],
        'consumer': ['WMT', 'HD', 'MCD'],
        'energy': ['XOM', 'CVX']
    }
    
    # Graphs
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
    
    # Better positioning with much more spacing
    pos = {}
    tech_center = (-2.5, 2.5)
    tech_radius = 2.5
    tech_angles = np.linspace(0, 2*np.pi, len(sectors['tech']), endpoint=False)
    for i, stock in enumerate(sectors['tech']):
        angle = tech_angles[i]
        pos[stock] = (tech_center[0] + tech_radius * np.cos(angle),
                     tech_center[1] + tech_radius * np.sin(angle))
    
    pos['JPM'] = (5.0, 3.0)
    pos['BAC'] = (7.0, 3.0)
    pos['JNJ'] = (-4.0, -3.5)
    pos['PFE'] = (-2.0, -3.5)
    pos['WMT'] = (1.0, -4.5)
    pos['HD'] = (-1.0, -4.5)
    pos['MCD'] = (0.0, -5.5)
    pos['XOM'] = (5.5, -3.5)
    pos['CVX'] = (7.5, -3.5)
    
    node_colors = {}
    for sector_name, sector_stocks in sectors.items():
        for stock in sector_stocks:
            node_colors[stock] = colors[sector_name]
    
    return stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors, colors


def _draw_clean_nodes(ax, stocks, pos, node_colors, size=0.3, fontsize=14):
    """Draw nodes cleanly."""
    for stock in stocks:
        if stock in pos:
            circle = Circle(pos[stock], size, color=node_colors.get(stock, '#95A5A6'), 
                          alpha=0.9, zorder=3, edgecolor='white', linewidth=4)
            ax.add_patch(circle)
            text = ax.text(pos[stock][0], pos[stock][1], stock, 
                          ha='center', va='center', fontsize=fontsize, 
                          fontweight='bold', color='white', zorder=4)
            text.set_path_effects([withStroke(linewidth=5, foreground='black', alpha=0.9)])


def create_figure7a_clean():
    """Figure 7a: Clean, minimal design."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors, colors = _get_clean_graph_data()
    
    fig = plt.figure(figsize=(22, 16), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.7, wspace=0.6, 
                          left=0.10, right=0.95, top=0.93, bottom=0.10)
    
    # Main graph
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor('#FAFAFA')
    
    # Draw edges - NO labels
    for edge in G_corr.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            ax_main.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                        [pos[edge[0]][1], pos[edge[1]][1]], 
                        color='#E74C3C', linewidth=weight*3 + 2, 
                        alpha=0.65, zorder=1, solid_capstyle='round')
    
    for edge in G_sector.edges():
        if edge[0] in pos and edge[1] in pos and edge not in G_corr.edges():
            ax_main.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                        [pos[edge[0]][1], pos[edge[1]][1]], 
                        color='#3498DB', linewidth=3, linestyle='--', 
                        dashes=(12, 6), alpha=0.6, zorder=1)
    
    for edge in G_fund.edges():
        if edge[0] in pos and edge[1] in pos and edge not in G_corr.edges() and edge not in G_sector.edges():
            weight = G_fund[edge[0]][edge[1]].get('weight', 0.7)
            ax_main.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                        [pos[edge[0]][1], pos[edge[1]][1]], 
                        color='#2ECC71', linewidth=weight*3 + 1.5, 
                        linestyle=':', dashes=(4, 6), alpha=0.6, zorder=1)
    
    _draw_clean_nodes(ax_main, stocks, pos, node_colors, size=0.35, fontsize=15)
    
    ax_main.set_xlim(-6.0, 9.0)
    ax_main.set_ylim(-6.5, 4.5)
    ax_main.set_title('Heterogeneous Stock Graph: Complete Multi-Relational Structure', 
                     fontsize=18, fontweight='bold', pad=35, color='#2C3E50')
    ax_main.axis('off')
    
    # Combined legend - single legend to avoid overlap
    legend_elements = [
        Line2D([0], [0], color='#E74C3C', linewidth=5, label='Correlation'),
        Line2D([0], [0], color='#3498DB', linestyle='--', linewidth=4, dashes=(12, 6), label='Sector'),
        Line2D([0], [0], color='#2ECC71', linestyle=':', linewidth=4, dashes=(4, 6), label='Fundamental'),
        mpatches.Patch(color=colors['tech'], label='Tech'),
        mpatches.Patch(color=colors['finance'], label='Finance'),
        mpatches.Patch(color=colors['healthcare'], label='Healthcare'),
        mpatches.Patch(color=colors['consumer'], label='Consumer'),
        mpatches.Patch(color=colors['energy'], label='Energy'),
    ]
    legend = ax_main.legend(handles=legend_elements, loc='upper right', fontsize=12, 
                           frameon=True, fancybox=True, shadow=True, 
                           framealpha=0.95, edgecolor='gray', facecolor='white',
                           bbox_to_anchor=(0.98, 0.98), ncol=2)
    legend.get_frame().set_linewidth(2)
    
    # Minimal stats - positioned in lower right corner to avoid overlap with graph
    stats_text = '14 Nodes\n34 Edges'
    stats_box = FancyBboxPatch((0.82, 0.02), 0.16, 0.09, 
                               boxstyle="round,pad=0.03", 
                               transform=ax_main.transAxes,
                               facecolor='white', edgecolor='#34495E', 
                               linewidth=3, alpha=0.95)
    ax_main.add_patch(stats_box)
    ax_main.text(0.90, 0.065, stats_text, transform=ax_main.transAxes, 
                fontsize=13, verticalalignment='center', horizontalalignment='center',
                color='#2C3E50', fontweight='bold')
    
    # Details
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor('#FAFAFA')
    tech_stocks = sectors['tech']
    tech_pos = {s: pos[s] for s in tech_stocks if s in pos}
    
    for edge in G_corr.edges():
        if edge[0] in tech_pos and edge[1] in tech_pos:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            ax1.plot([tech_pos[edge[0]][0], tech_pos[edge[1]][0]], 
                    [tech_pos[edge[0]][1], tech_pos[edge[1]][1]], 
                    color='#E74C3C', linewidth=weight*4 + 2, alpha=0.75, zorder=1)
    
    for stock in tech_stocks:
        if stock in tech_pos:
            circle = Circle(tech_pos[stock], 0.25, color=node_colors[stock], 
                          alpha=0.95, zorder=3, edgecolor='white', linewidth=4)
            ax1.add_patch(circle)
            text = ax1.text(tech_pos[stock][0], tech_pos[stock][1], stock, 
                          ha='center', va='center', fontsize=12, 
                          fontweight='bold', color='white', zorder=4)
            text.set_path_effects([withStroke(linewidth=5, foreground='black', alpha=0.9)])
    
    tech_x = [tech_pos[s][0] for s in tech_stocks if s in tech_pos]
    tech_y = [tech_pos[s][1] for s in tech_stocks if s in tech_pos]
    ax1.set_xlim(min(tech_x) - 2.0, max(tech_x) + 2.0)
    ax1.set_ylim(min(tech_y) - 2.0, max(tech_y) + 2.0)
    ax1.set_title('Technology Cluster', fontsize=15, fontweight='bold', pad=25, color='#2C3E50')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor('#FAFAFA')
    _draw_clean_nodes(ax2, stocks, pos, node_colors, size=0.28, fontsize=12)
    
    cross_sector_edges = []
    for edge in G_corr.edges():
        s1, s2 = edge[0], edge[1]
        s1_sector = next((k for k, v in sectors.items() if s1 in v), None)
        s2_sector = next((k for k, v in sectors.items() if s2 in v), None)
        if s1_sector != s2_sector and s1_sector and s2_sector:
            cross_sector_edges.append(edge)
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            ax2.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                    [pos[edge[0]][1], pos[edge[1]][1]], 
                    color='#E74C3C', linewidth=weight*4 + 3, alpha=0.85, zorder=1)
    
    ax2.set_xlim(-6.0, 9.0)
    ax2.set_ylim(-6.5, 4.5)
    ax2.set_title(f'Cross-Sector ({len(cross_sector_edges)} edges)', 
                 fontsize=15, fontweight='bold', pad=25, color='#2C3E50')
    ax2.axis('off')
    
    plt.suptitle('Figure 7a: Heterogeneous Graph Structure Overview', 
                fontsize=20, fontweight='bold', y=0.97, color='#2C3E50')
    plt.savefig(FIGS_DIR / 'figure7a_graph_structure_overview.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7a_graph_structure_overview.png (clean)")


def create_figure7b_clean():
    """Figure 7b: Clean correlation edges."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors, colors = _get_clean_graph_data()
    
    fig = plt.figure(figsize=(22, 16), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.7, wspace=0.6, 
                          left=0.10, right=0.95, top=0.93, bottom=0.10)
    
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor('#FAFAFA')
    
    edge_weights = []
    for edge in G_corr.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_corr[edge[0]][edge[1]].get('weight', 0.7)
            edge_weights.append((edge, weight))
            ax_main.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                        [pos[edge[0]][1], pos[edge[1]][1]], 
                        color='#E74C3C', linewidth=weight*4 + 3, 
                        alpha=0.7, zorder=1, solid_capstyle='round')
    
    _draw_clean_nodes(ax_main, stocks, pos, node_colors, size=0.35, fontsize=15)
    
    ax_main.set_xlim(-6.0, 9.0)
    ax_main.set_ylim(-6.5, 4.5)
    ax_main.set_title('Rolling Correlation Edges: Dynamic Price Co-Movements', 
                     fontsize=18, fontweight='bold', pad=35, color='#2C3E50')
    ax_main.axis('off')
    
    # Minimal description
    desc_text = 'Dynamic edges\n30-day rolling\nTop-K=10\nDaily update'
    desc_box = FancyBboxPatch((0.02, 0.02), 0.16, 0.12, 
                             boxstyle="round,pad=0.03", 
                             transform=ax_main.transAxes,
                             facecolor='#FFF3E0', edgecolor='#E74C3C', 
                             linewidth=3, alpha=0.95)
    ax_main.add_patch(desc_box)
    ax_main.text(0.10, 0.08, desc_text, transform=ax_main.transAxes, 
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                color='#2C3E50', fontweight='bold')
    
    # Top 5
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor('#FAFAFA')
    sorted_edges = sorted(edge_weights, key=lambda x: x[1], reverse=True)[:5]
    strong_stocks = set()
    for (edge, weight) in sorted_edges:
        strong_stocks.add(edge[0])
        strong_stocks.add(edge[1])
        ax1.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                [pos[edge[0]][1], pos[edge[1]][1]], 
                color='#E74C3C', linewidth=weight*5 + 4, alpha=0.8, zorder=1)
    
    for stock in strong_stocks:
        if stock in pos:
            circle = Circle(pos[stock], 0.30, color=node_colors[stock], 
                          alpha=0.95, zorder=3, edgecolor='white', linewidth=4)
            ax1.add_patch(circle)
            text = ax1.text(pos[stock][0], pos[stock][1], stock, 
                          ha='center', va='center', fontsize=13, 
                          fontweight='bold', color='white', zorder=4)
            text.set_path_effects([withStroke(linewidth=5, foreground='black', alpha=0.9)])
    
    for stock in stocks:
        if stock in pos and stock not in strong_stocks:
            circle = Circle(pos[stock], 0.20, color='#BDC3C7', 
                          alpha=0.3, zorder=2, edgecolor='gray', linewidth=1)
            ax1.add_patch(circle)
            ax1.text(pos[stock][0], pos[stock][1], stock, 
                    ha='center', va='center', fontsize=10, 
                    color='gray', zorder=3, alpha=0.5)
    
    ax1.set_xlim(-6.0, 9.0)
    ax1.set_ylim(-6.5, 4.5)
    ax1.set_title('Top 5 Strongest Correlations', 
                 fontsize=15, fontweight='bold', pad=25, color='#2C3E50')
    ax1.axis('off')
    
    # Distribution
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor('white')
    weights = [w for _, w in edge_weights]
    ax2.hist(weights, bins=12, color='#E74C3C', edgecolor='darkred', 
            linewidth=2, alpha=0.7)
    ax2.axvline(x=np.mean(weights), color='blue', linestyle='--', 
               linewidth=3, label=f'Mean: {np.mean(weights):.3f}')
    ax2.set_xlabel('Correlation Coefficient', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax2.set_title('Correlation Distribution', fontsize=15, fontweight='bold', pad=25, color='#2C3E50')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Figure 7b: Rolling Correlation Edges Analysis', 
                fontsize=20, fontweight='bold', y=0.97, color='#2C3E50')
    plt.savefig(FIGS_DIR / 'figure7b_correlation_edges.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7b_correlation_edges.png (clean)")


def create_figure7c_clean():
    """Figure 7c: Clean sector edges."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors, colors = _get_clean_graph_data()
    
    fig = plt.figure(figsize=(22, 16), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.7, wspace=0.6, 
                          left=0.10, right=0.95, top=0.93, bottom=0.10)
    
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor('#FAFAFA')
    
    for edge in G_sector.edges():
        if edge[0] in pos and edge[1] in pos:
            ax_main.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                        [pos[edge[0]][1], pos[edge[1]][1]], 
                        color='#3498DB', linewidth=4, linestyle='--', 
                        dashes=(12, 6), alpha=0.7, zorder=1)
    
    _draw_clean_nodes(ax_main, stocks, pos, node_colors, size=0.35, fontsize=15)
    
    ax_main.set_xlim(-6.0, 9.0)
    ax_main.set_ylim(-6.5, 4.5)
    ax_main.set_title('Sector/Industry Edges: Static Domain Knowledge', 
                     fontsize=18, fontweight='bold', pad=35, color='#2C3E50')
    ax_main.axis('off')
    
    desc_text = 'Static edges\nIndustry-based\nBinary (same=1)\nNever updated'
    desc_box = FancyBboxPatch((0.02, 0.02), 0.16, 0.12, 
                             boxstyle="round,pad=0.03", 
                             transform=ax_main.transAxes,
                             facecolor='#E3F2FD', edgecolor='#3498DB', 
                             linewidth=3, alpha=0.95)
    ax_main.add_patch(desc_box)
    ax_main.text(0.10, 0.08, desc_text, transform=ax_main.transAxes, 
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                color='#2C3E50', fontweight='bold')
    
    # Sector clusters
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor('#FAFAFA')
    
    for edge in G_sector.edges():
        if edge[0] in pos and edge[1] in pos:
            ax1.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                    [pos[edge[0]][1], pos[edge[1]][1]], 
                    color='#3498DB', linewidth=4, linestyle='--', 
                    dashes=(12, 6), alpha=0.7, zorder=1)
    
    sector_centers = {}
    for sector_name, sector_stocks in sectors.items():
        if len(sector_stocks) > 0:
            x_coords = [pos[s][0] for s in sector_stocks if s in pos]
            y_coords = [pos[s][1] for s in sector_stocks if s in pos]
            if x_coords and y_coords:
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                sector_centers[sector_name] = (center_x, center_y)
                ax1.text(center_x, center_y + 1.5, sector_name.upper(), 
                        ha='center', va='center', fontsize=15, 
                        fontweight='bold', color=colors[sector_name],
                        bbox=dict(boxstyle='round,pad=0.8', 
                                facecolor='white', alpha=0.95,
                                edgecolor=colors[sector_name], linewidth=3),
                        zorder=6)
    
    _draw_clean_nodes(ax1, stocks, pos, node_colors, size=0.28, fontsize=12)
    
    ax1.set_xlim(-6.0, 9.0)
    ax1.set_ylim(-6.5, 4.5)
    ax1.set_title('Sector Clusters', fontsize=15, fontweight='bold', pad=25, color='#2C3E50')
    ax1.axis('off')
    
    # Statistics table
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor('white')
    ax2.axis('off')
    
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
    
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 3.0)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_edgecolor('white')
        table[(0, i)].set_linewidth(2.5)
    
    for i in range(1, len(table_data)):
        for j in range(4):
            table[(i, j)].set_edgecolor('#BDC3C7')
            table[(i, j)].set_linewidth(1.5)
            if j == 0:
                sector_name = table_data[i][0].lower()
                if sector_name in colors:
                    table[(i, j)].set_facecolor(colors[sector_name] + '40')
    
    ax2.set_title('Sector Statistics', fontsize=15, fontweight='bold', pad=30, color='#2C3E50')
    
    plt.suptitle('Figure 7c: Sector/Industry Edges Analysis', 
                fontsize=20, fontweight='bold', y=0.97, color='#2C3E50')
    plt.savefig(FIGS_DIR / 'figure7c_sector_edges.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7c_sector_edges.png (clean)")


def create_figure7d_clean():
    """Figure 7d: Clean fundamental edges."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors, colors = _get_clean_graph_data()
    
    fig = plt.figure(figsize=(22, 16), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.7, wspace=0.6, 
                          left=0.10, right=0.95, top=0.93, bottom=0.10)
    
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor('#FAFAFA')
    
    edge_weights = []
    for edge in G_fund.edges():
        if edge[0] in pos and edge[1] in pos:
            weight = G_fund[edge[0]][edge[1]].get('weight', 0.7)
            edge_weights.append((edge, weight))
            ax_main.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                        [pos[edge[0]][1], pos[edge[1]][1]], 
                        color='#2ECC71', linewidth=weight*4 + 3, 
                        linestyle=':', dashes=(4, 6), alpha=0.7, zorder=1)
    
    _draw_clean_nodes(ax_main, stocks, pos, node_colors, size=0.35, fontsize=15)
    
    ax_main.set_xlim(-6.0, 9.0)
    ax_main.set_ylim(-6.5, 4.5)
    ax_main.set_title('Fundamental Similarity Edges: Long-Term Value Alignment', 
                     fontsize=18, fontweight='bold', pad=35, color='#2C3E50')
    ax_main.axis('off')
    
    desc_text = 'Static edges\nCosine similarity\nP/E, ROE features\nThreshold: >0.7\nQuarterly'
    desc_box = FancyBboxPatch((0.02, 0.02), 0.16, 0.14, 
                             boxstyle="round,pad=0.03", 
                             transform=ax_main.transAxes,
                             facecolor='#E8F5E9', edgecolor='#2ECC71', 
                             linewidth=3, alpha=0.95)
    ax_main.add_patch(desc_box)
    ax_main.text(0.10, 0.09, desc_text, transform=ax_main.transAxes, 
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                color='#2C3E50', fontweight='bold')
    
    # Network
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor('#FAFAFA')
    
    for edge, weight in edge_weights:
        if edge[0] in pos and edge[1] in pos:
            ax1.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                    [pos[edge[0]][1], pos[edge[1]][1]], 
                    color='#2ECC71', linewidth=weight*4 + 3, 
                    linestyle=':', dashes=(4, 6), alpha=0.75, zorder=1)
    
    _draw_clean_nodes(ax1, stocks, pos, node_colors, size=0.28, fontsize=12)
    
    ax1.set_xlim(-6.0, 9.0)
    ax1.set_ylim(-6.5, 4.5)
    ax1.set_title('Fundamental Network', fontsize=15, fontweight='bold', pad=25, color='#2C3E50')
    ax1.axis('off')
    
    # Distribution
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor('white')
    weights = [w for _, w in edge_weights]
    ax2.hist(weights, bins=10, color='#2ECC71', edgecolor='darkgreen', 
            linewidth=2, alpha=0.7)
    ax2.axvline(x=0.7, color='red', linestyle='--', linewidth=3, label='Threshold: 0.7')
    ax2.axvline(x=np.mean(weights), color='blue', linestyle='--', 
               linewidth=3, label=f'Mean: {np.mean(weights):.3f}')
    ax2.set_xlabel('Similarity Score', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax2.set_title('Similarity Distribution', fontsize=15, fontweight='bold', pad=25, color='#2C3E50')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Figure 7d: Fundamental Similarity Edges Analysis', 
                fontsize=20, fontweight='bold', y=0.97, color='#2C3E50')
    plt.savefig(FIGS_DIR / 'figure7b_correlation_edges.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7d_fundamental_edges.png (clean)")


def create_figure7e_clean():
    """Figure 7e: Clean edge comparison."""
    stocks, sectors, G_corr, G_sector, G_fund, pos, node_colors, colors = _get_clean_graph_data()
    
    fig = plt.figure(figsize=(24, 16), facecolor='white')
    gs = fig.add_gridspec(2, 3, hspace=0.7, wspace=0.6, 
                          left=0.08, right=0.96, top=0.93, bottom=0.10)
    
    edge_types = [
        (G_corr, '#E74C3C', 'Rolling Correlation', 'Dynamic'),
        (G_sector, '#3498DB', 'Sector/Industry', 'Static'),
        (G_fund, '#2ECC71', 'Fundamental Similarity', 'Static')
    ]
    
    for idx, (G, edge_color, edge_name, edge_type) in enumerate(edge_types):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor('#FAFAFA')
        
        if edge_name == 'Rolling Correlation':
            for edge in G.edges():
                if edge[0] in pos and edge[1] in pos:
                    weight = G[edge[0]][edge[1]].get('weight', 0.7)
                    ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                           [pos[edge[0]][1], pos[edge[1]][1]], 
                           color=edge_color, linewidth=weight*4 + 2.5, 
                           alpha=0.7, zorder=1, solid_capstyle='round')
        elif edge_name == 'Sector/Industry':
            for edge in G.edges():
                if edge[0] in pos and edge[1] in pos:
                    ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                           [pos[edge[0]][1], pos[edge[1]][1]], 
                           color=edge_color, linewidth=4, linestyle='--', 
                           dashes=(12, 6), alpha=0.7, zorder=1)
        else:
            for edge in G.edges():
                if edge[0] in pos and edge[1] in pos:
                    weight = G[edge[0]][edge[1]].get('weight', 0.7)
                    ax.plot([pos[edge[0]][0], pos[edge[1]][0]], 
                           [pos[edge[0]][1], pos[edge[1]][1]], 
                           color=edge_color, linewidth=weight*4 + 2.5, 
                           linestyle=':', dashes=(4, 6), alpha=0.7, zorder=1)
        
        _draw_clean_nodes(ax, stocks, pos, node_colors, size=0.30, fontsize=13)
        
        ax.set_xlim(-6.0, 9.0)
        ax.set_ylim(-6.5, 4.5)
        ax.set_title(f'{edge_name}\n({edge_type})', 
                    fontsize=16, fontweight='bold', pad=30, color='#2C3E50')
        ax.axis('off')
    
    # Comparison table
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.set_facecolor('white')
    ax_stats.axis('off')
    
    comparison_data = [
        ['Edge Type', 'Count', 'Type', 'Updates', 'Weighted', 'Purpose'],
        ['Rolling Correlation', '12', 'Dynamic', 'Daily', 'Yes', 'Short-term co-movements'],
        ['Sector/Industry', '13', 'Static', 'Never', 'No', 'Industry-level factors'],
        ['Fundamental Similarity', '9', 'Static', 'Quarterly', 'Yes', 'Long-term value alignment'],
        ['Total', '34', 'Mixed', 'Mixed', 'Mixed', 'Multi-relational learning']
    ]
    
    table = ax_stats.table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                           cellLoc='center', loc='center',
                           colWidths=[0.24, 0.12, 0.12, 0.12, 0.12, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 4.0)
    
    for i in range(6):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_edgecolor('white')
        table[(0, i)].set_linewidth(3)
    
    row_colors = ['#FFEBEE', '#E3F2FD', '#E8F5E9', '#FFF9E6']
    for i in range(1, len(comparison_data)):
        for j in range(6):
            table[(i, j)].set_edgecolor('#BDC3C7')
            table[(i, j)].set_linewidth(2)
            if i < len(comparison_data) - 1:
                table[(i, j)].set_facecolor(row_colors[i-1])
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
                table[(i, j)].set_text_props(weight='bold')
    
    ax_stats.set_title('Edge Type Comparison', 
                      fontsize=17, fontweight='bold', color='#2C3E50', pad=40)
    
    plt.suptitle('Figure 7e: Edge Type Comparison and Analysis', 
                fontsize=20, fontweight='bold', y=0.97, color='#2C3E50')
    plt.savefig(FIGS_DIR / 'figure7e_edge_comparison.png', 
               bbox_inches='tight', dpi=300, facecolor='white', pad_inches=0.2)
    plt.close()
    print(f"✅ Created: figure7e_edge_comparison.png (clean)")


if __name__ == "__main__":
    print("Generating clean Figure 7 series...")
    create_figure7a_clean()
    create_figure7b_clean()
    create_figure7c_clean()
    create_figure7d_clean()
    create_figure7e_clean()
    print("\n✅ All clean Figure 7 charts generated!")

