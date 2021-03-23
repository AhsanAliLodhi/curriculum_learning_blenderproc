import numpy as np
import cv2


# matr = [1,2,3,4,5,6]

# res = [[x] for x in matr]



# data = np.load('/dev/shm/blender_proc_aff15cf5f6f74700a25abbf5620b2989 (copy)/segmap_0000.npy')
# cv2.imshow("0", data[:,:,0])
# cv2.imshow("1", data[:,:,1])
# #cv2.imshow("2", data[:,:,2])
# cv2.waitKey(0)
def riverSizes(matrix):
		sizes = []
		visited = [[True if value > 0 else False for value in row] for row in matrix]
		for i in range(len(matrix)):
				for j in range(len(matrix[i])):
						if visited[i][j]:
							continue
						traverseNode(i, j, matrix, visited, sizes)
		return sizes
	
def traverseNode(i, j, matrix, visited, sizes):
		currentRiverSize = 0
		nodesToExplore = [[i,j]]
		while len(nodesToExplore):
				currentNode = nodesToExplore.pop()
				i = currentNode[0]
				j = currentNode[1]
				if visited[i][j]:
					continue
				visited[i][j] = True
				if matrix[i][j] == 0:
						continue
				currentRiverSize += 1
				unvisitedNeighbors = getUnvisitedNeighbours(i, j, matrix, visited)
				
				for neighbor in unvisitedNeighbors:
						nodesToExplore.append(neighbor)
				
		if currentRiverSize > 0:
				sizes.append(currentRiverSize)

				
def getUnvisitedNeighbours(i, j, matix, visited):
		unvisitedNeighbors = []
		if i > 0 and not visited[i-1][j]:
				unvisitedNeighbors.append([i-1,j])
				
		if i < len(matrix)-1 and not visited[i+1][j]:
				unvisitedNeighbors.append([i+1,j])
				
		if j > 0 and not visited[i][j-1]:
				unvisitedNeighbors.append([i,j-1])
				
		if j < len(matrix[0])-1 and not visited[i][j+1]:
				unvisitedNeighbors.append([i,j+1])		
			
		return unvisitedNeighbors

	
matrix = [
	[1,0,0,1,0],
	[1,0,1,0,0],
	[0,0,1,0,1],
	[1,0,1,0,1],
	[1,0,1,1,0],
]	
sizes = riverSizes(matrix)
print(sizes)
