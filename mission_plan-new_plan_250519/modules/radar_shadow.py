"""
레이더 음영(Radar Shadow) 계산 모듈
Terrain Masking & Line-of-Sight (LoS) 알고리즘
"""
import math

def check_line_of_sight(
    radar_pos: tuple,   # (lat, lon, alt_m)
    aircraft_pos: tuple, # (lat, lon, alt_m)
    terrain_loader,
    samples: int = 15    # 샘플링 횟수 (높을수록 정밀하지만 느림)
) -> bool:
    """
    레이더와 항공기 사이의 가시선(LoS) 검사
    
    Returns:
        True: 탐지됨 (장애물 없음)
        False: 탐지 불가 (지형에 가려짐 = Shadow)
    """
    r_lat, r_lon, r_alt = radar_pos
    a_lat, a_lon, a_alt = aircraft_pos

    # 1. 항공기가 레이더보다 현저히 낮고(MSL 기준 50m 미만) 음영 간주
    # (AGL이 아닌 MSL 비교이므로 레이더 위에 있는 경우는 제외)
    if a_alt < r_alt - 50:
        return False  # 레이더보다 훨씬 낮은 초저고도 → 음영 간주

    # 2. 샘플링을 통한 LoS 체크
    for i in range(1, samples + 1):
        ratio = i / (samples + 1)
        
        # 중간 지점 좌표 계산 (선형 보간)
        mid_lat = r_lat + (a_lat - r_lat) * ratio
        mid_lon = r_lon + (a_lon - r_lon) * ratio
        
        # 중간 지점의 가시선(Line) 높이 계산
        line_alt = r_alt + (a_alt - r_alt) * ratio
        
        # 중간 지점의 실제 지형(Terrain) 높이 조회
        terrain_alt = terrain_loader.get_elevation(mid_lat, mid_lon)
        
        # 지형이 가시선보다 높으면 -> 레이더 전파 차단됨! (Shadow)
        # 약간의 오차를 고려해 지형이 10m 이상 더 높아야 차단으로 인정
        if terrain_alt > line_alt + 10:
            return False # 차단됨 (은폐 성공)
            
    return True # 차단되지 않음 (탐지됨)